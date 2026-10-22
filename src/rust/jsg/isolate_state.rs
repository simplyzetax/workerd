// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

//! Release of persistent handles, and the liveness of their isolate.
//!
//! A [`v8::Global`](crate::v8::Global) outlives every scope, and is dropped wherever the Rust
//! value owning it goes away. Two things about its isolate decide what the drop may do:
//!
//! * **Whether the isolate is locked by this thread.** `v8::Global::Reset` edits the isolate's
//!   free list of handle nodes without synchronization, as does every handle creation on the
//!   thread holding the lock. A reset without the lock is a data race that can hand a node out
//!   twice or unlink a block of roots.
//! * **Whether the isolate is still alive.** Nothing ties a `Global` to its isolate's
//!   lifetime: one can be dropped after `v8::Isolate::Dispose`, for example when the Rust
//!   object holding it is owned by something that outlives the worker. `Dispose` frees every
//!   global and traced handle of the isolate in bulk, so a reset after it touches freed
//!   memory.
//!
//! The [`Realm`](crate::Realm) therefore owns an [`IsolateState`], shared through an `Arc`
//! with every `Global` of its isolate. [`IsolateState::release`] resets a dropped `Global`'s
//! handles at once under the lock, queues them otherwise, and leaves them to `Dispose` once
//! the state is closed. The queue is replayed by [`IsolateState::drain`] through the
//! `IsolateBase` deferred action hook, which `IsolateBase::Lock` calls whenever it locks the
//! isolate, the same point at which C++ JSG applies its own deferred destruction queue. A bare
//! `v8::Locker`, or relocking after a `v8::Unlocker`, does not drain the queue; the handles
//! then wait for the next `IsolateBase::Lock`. The `Realm` closes the state when it is
//! dropped, under the isolate lock, before the isolate is disposed.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::Isolate;
use crate::scope::scope_owner;
use crate::v8;
use crate::v8::ffi;

/// What the handles of an isolate need to know about it after their scopes are gone.
pub struct IsolateState {
    /// Compared with the isolate whose scopes are open on the current thread, and dereferenced
    /// only while the isolate is known to be alive: while `deferred` is locked and `closed` is
    /// not set, since [`close`](Self::close) sets `closed` under that same lock, before the
    /// isolate is disposed.
    isolate: *mut ffi::Isolate,
    /// Set by [`close`](Self::close), under the isolate lock, before the isolate is disposed.
    closed: AtomicBool,
    /// Strong handles of `Global`s that were dropped without the isolate lock, waiting for
    /// the next lock.
    deferred: Mutex<Vec<ffi::Global>>,
}

// SAFETY: `isolate` is a raw pointer that is only compared, or dereferenced for
// `v8::Locker::IsLocked`, which any thread may call, while the isolate is alive (see the
// field). The other fields synchronize themselves.
#[expect(unsafe_code, reason = "shares the raw isolate pointer between threads")]
unsafe impl Send for IsolateState {}
// SAFETY: as for `Send`.
#[expect(unsafe_code, reason = "shares the raw isolate pointer between threads")]
unsafe impl Sync for IsolateState {}

#[expect(
    unsafe_code,
    reason = "resets V8 handles and asks V8 whether the isolate is locked"
)]
impl IsolateState {
    pub(crate) fn new(isolate: v8::IsolatePtr) -> Arc<Self> {
        Arc::new(Self {
            isolate: isolate.as_ffi(),
            closed: AtomicBool::new(false),
            deferred: Mutex::new(Vec::new()),
        })
    }

    /// Whether the isolate is about to be disposed, or already has been. Its handles are then
    /// freed with it, and must not be touched.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Releases the handles of a dropped `Global`.
    ///
    /// Under the isolate lock the handles are reset at once. Without it the strong handle is
    /// queued, and [`drain`](Self::drain) resets it the next time the isolate is locked, so
    /// until then the handle keeps its value alive. Once the isolate is closed the handles
    /// are left to `v8::Isolate::Dispose`.
    pub(crate) fn release(&self, strong: ffi::Global, traced: ffi::TracedReference) {
        if (strong.ptr == 0 && traced.ptr == 0) || self.is_closed() {
            return;
        }
        if scope_owner::current() == self.isolate {
            // A scope of the isolate is open on this thread, so this thread holds its lock.
            reset(strong, traced);
            return;
        }
        let mut deferred = self.lock_deferred();
        if self.is_closed() {
            return;
        }
        // `close` takes `deferred` before it sets `closed`, so the isolate is alive until the
        // guard is dropped.
        // SAFETY: the isolate is alive (see above), and `v8::Locker::IsLocked` may be called
        // from any thread.
        if unsafe { ffi::isolate_is_locked(self.isolate) } {
            drop(deferred);
            reset(strong, traced);
            return;
        }
        // Only GC installs a traced reference, on a `Global` inside a resource, and V8 then
        // writes into the strong handle's slot when its value dies, so such a `Global` must
        // not be dropped without the lock while its slot moves with it. A resource drops its
        // handles under the lock, as its own destruction is deferred to it. The traced handle
        // needs no release: V8 reclaims an unreferenced traced node at the next GC.
        debug_assert_eq!(
            traced.ptr, 0,
            "a Global in traced mode was dropped without the isolate lock"
        );
        deferred.push(strong);
    }

    /// Resets the handles queued by [`release`](Self::release). The deferred action hook calls
    /// it whenever `IsolateBase::Lock` locks the isolate.
    pub(crate) fn drain(&self, _isolate: &mut Isolate) {
        // The handles are reset outside the mutex, which only guards the queue itself.
        let batch = std::mem::take(&mut *self.lock_deferred());
        for strong in batch {
            reset_strong(strong);
        }
    }

    /// The number of handles waiting for the next lock.
    pub(crate) fn deferred_count(&self) -> usize {
        self.lock_deferred().len()
    }

    /// Records that the isolate is about to be disposed, and resets the queued handles while
    /// it still can. Every handle of the isolate that is dropped from now on is left to
    /// `Dispose`.
    ///
    /// The `Isolate` reference proves that the caller holds the isolate lock; no handle of the
    /// isolate can be reset concurrently.
    pub(crate) fn close(&self, _isolate: &mut Isolate) {
        let batch = {
            let mut deferred = self.lock_deferred();
            self.closed.store(true, Ordering::Release);
            std::mem::take(&mut *deferred)
        };
        for strong in batch {
            reset_strong(strong);
        }
    }

    fn lock_deferred(&self) -> MutexGuard<'_, Vec<ffi::Global>> {
        // The mutex is only held around `Vec` operations, so a poisoned lock still guards a
        // consistent queue.
        self.deferred.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Resets a `Global`'s handles. The isolate must be locked by this thread.
#[expect(unsafe_code, reason = "resets V8 handles")]
fn reset(strong: ffi::Global, mut traced: ffi::TracedReference) {
    reset_strong(strong);
    if traced.ptr != 0 {
        // SAFETY: the traced handle is valid, and the isolate is locked (guaranteed by caller).
        unsafe { ffi::traced_reference_reset(&mut traced) };
    }
}

/// Resets a strong handle. The isolate must be locked by this thread.
#[expect(unsafe_code, reason = "resets a V8 handle")]
fn reset_strong(mut strong: ffi::Global) {
    // SAFETY: the handle is valid and the isolate is locked (guaranteed by caller). `strong`
    // is not moved while pinned.
    unsafe { ffi::global_reset(Pin::new_unchecked(&mut strong)) };
}
