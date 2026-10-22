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
//!
//! # Boxed handles
//!
//! A `Global` that GC has visited keeps its handles in a box (see
//! [`Handles`](crate::v8::Handles)), because V8 writes into the slot of a weak handle when its
//! value dies, during any GC. GC runs on the thread holding the isolate lock, so a box is only
//! read under the lock, and freed only once its strong handle has been reset, which is when V8
//! forgets the slot. A release without the lock queues the box. After the state is closed the
//! handles are no longer reset, but GC can still run before the isolate is disposed: during
//! `IsolateBase::dropWrappers`, while `v8::Isolate::Dispose` tears the heap down, or whenever
//! V8 flags ask for it. So the box of a `Global` dropped then is parked, and freed once
//! `~IsolateBase` reports, through a hook the `Realm` set, that the isolate is disposed.
//!
//! # Traced handles
//!
//! When GC visits a `Global` inside a resource that is only reachable through its wrapper, it
//! makes the strong handle weak and installs a `v8::TracedReference`, which the resource's
//! trace marks in every GC that can free traced nodes: every full mark-compact, and every
//! minor mark-sweep when cppgc's young generation is enabled. V8 frees the traced nodes that
//! such a GC did not mark, in its atomic pause (`TracedHandles::ResetDeadNodes`, or
//! `ResetYoungDeadNodes` for a minor mark-sweep). A `Global` taken out of its resource
//! (`Cell::take`, `mem::replace`) is traced by nothing until it is put back, and nothing tells
//! it that it moved, the way a C++ move constructor would; after the next such GC it holds a
//! pointer to a freed node. Resetting that node would free it a second time, and tracing it
//! would mark whatever V8 reused it for.
//!
//! The state therefore counts the garbage collections that can free traced nodes (see
//! [`IsolateState::gc_epoch`]), and each traced handle records the epoch through which its
//! node is known to be live (see [`TracedHandle`]). A node that GC may have freed is
//! forgotten: the pointer is cleared without a reset, and V8 reclaims the node at the next
//! such GC if it is still there. A node known to be live is reset when its `Global` is
//! dropped, as C++ JSG's `jsg::Data` does. Left alone it would occupy its slot until the next
//! such GC, and while its value is in the young generation it would keep the value alive,
//! since scavenges treat young traced nodes as roots.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::Isolate;
use crate::scope::scope_owner;
use crate::v8;
use crate::v8::HandleSlot;
use crate::v8::Handles;
use crate::v8::ffi;

/// What the handles of an isolate need to know about it after their scopes are gone.
pub struct IsolateState {
    /// Compared with the isolate whose scopes are open on the current thread, and dereferenced
    /// only while the isolate is known to be alive: while `queues` is locked and `closed` is
    /// not set, since [`close`](Self::close) sets `closed` under that same lock, before the
    /// isolate is disposed.
    isolate: *mut ffi::Isolate,
    /// Set by [`close`](Self::close), under the isolate lock, before the isolate is disposed.
    closed: AtomicBool,
    /// See [`gc_epoch`](Self::gc_epoch). Only accessed by the thread holding the isolate lock,
    /// whose hand-over orders the accesses.
    gc_epoch: AtomicU64,
    queues: Mutex<Queues>,
}

#[derive(Default)]
struct Queues {
    /// Handles of `Global`s that were dropped without the isolate lock, waiting for the next
    /// lock.
    deferred: Vec<Handles>,
    /// Boxes of `Global`s that were dropped after the state was closed, waiting for the
    /// isolate to be disposed.
    #[expect(
        clippy::vec_box,
        reason = "V8 may write into a box until the isolate is disposed"
    )]
    parked: Vec<Box<HandleSlot>>,
    /// Set once the isolate is disposed, after which boxes are freed at once.
    disposed: bool,
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
            gc_epoch: AtomicU64::new(0),
            queues: Mutex::new(Queues::default()),
        })
    }

    /// Whether `isolate` is the isolate of this state. Only compares the pointers.
    pub(crate) fn belongs_to(&self, isolate: *mut ffi::Isolate) -> bool {
        self.isolate == isolate
    }

    /// Whether the isolate is locked by this thread. False once the state is closed.
    pub(crate) fn is_locked_by_this_thread(&self) -> bool {
        if self.is_closed() {
            return false;
        }
        if scope_owner::current() == self.isolate {
            // A scope of the isolate is open on this thread, so this thread holds its lock.
            return true;
        }
        let _queues = self.lock_queues();
        // `close` takes the queue's mutex before it sets `closed`, so the isolate is alive
        // until the guard is dropped. If this thread holds the lock, nothing can close the
        // state before it lets go of the lock.
        // SAFETY: the isolate is alive (see above), and `v8::Locker::IsLocked` may be called
        // from any thread.
        !self.is_closed() && unsafe { ffi::isolate_is_locked(self.isolate) }
    }

    /// Whether the isolate is about to be disposed, or already has been. Its handles are then
    /// freed with it, and must not be touched.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// The GC epoch: twice the number of garbage collections so far that can free traced
    /// nodes, plus one while such a collection is in progress. Only meaningful to the thread
    /// holding the isolate lock.
    ///
    /// Those collections are every full mark-compact, and every minor mark-sweep, which frees
    /// the young traced nodes it did not mark when cppgc's young generation is enabled.
    /// Scavenges free only droppable traced nodes, which Rust does not create. The `Realm`
    /// has V8 call [`note_gc`](Self::note_gc) in the prologue and in the epilogue of each
    /// such collection, and V8 frees traced nodes between the two.
    ///
    /// V8 skips the callbacks of a collection that starts inside another collection's
    /// callbacks, so such a collection goes uncounted. It can only free a node that the
    /// enclosing collection marked if code running in that collection's callbacks or
    /// finalizers took the node's `Global` out of its resource.
    pub(crate) fn gc_epoch(&self) -> u64 {
        self.gc_epoch.load(Ordering::Relaxed)
    }

    /// Records the start or the end of a garbage collection that can free traced nodes.
    pub(crate) fn note_gc(&self) {
        self.gc_epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// The epoch before which GC may have freed a traced node that a GC visit is looking at;
    /// see [`TracedHandle`]. That is the current epoch, except while the collection in
    /// progress is marking: V8 frees traced nodes only once marking is complete, so a node
    /// known to be live until the collection started is still there. cppgc reports marking
    /// until it leaves the atomic pause, which is after V8 freed the nodes, but GC visits are
    /// the only Rust code that looks at traced handles while cppgc reports marking.
    ///
    /// The exception is an optimisation. Without it, every visit by a collection's marking
    /// would forget a live node and install a new one, which is still correct, since the
    /// marking does not reach the forgotten node and the same collection frees it.
    ///
    /// The isolate must be locked by this thread, and the state must not be closed.
    pub(crate) fn stale_before_visit(&self) -> u64 {
        let epoch = self.gc_epoch();
        // SAFETY: the isolate is alive and locked by this thread (guaranteed by caller).
        if epoch % 2 == 1 && unsafe { ffi::isolate_is_marking(self.isolate) } {
            epoch - 1
        } else {
            epoch
        }
    }

    /// Releases the handles of a dropped `Global`.
    ///
    /// Under the isolate lock the strong handle is reset at once, and so is the traced handle
    /// if its node is known to be live (see the module documentation). Without the lock the
    /// handles are queued, and [`drain`](Self::drain) resets them the next time the isolate is
    /// locked, so until then the strong handle keeps its value alive. Once the isolate is
    /// closed the handles are left to `v8::Isolate::Dispose`, and a box is parked until then.
    pub(crate) fn release(&self, handles: Handles) {
        // GC never writes into an inline handle, so it may be read here. A box is read only
        // under the isolate lock.
        if matches!(handles, Handles::Inline(ffi::Global { ptr: 0 })) {
            return;
        }
        if self.is_closed() {
            self.park(handles);
            return;
        }
        if scope_owner::current() == self.isolate {
            // A scope of the isolate is open on this thread, so this thread holds its lock.
            self.reset(handles);
            return;
        }
        let mut queues = self.lock_queues();
        if self.is_closed() {
            drop(queues);
            self.park(handles);
            return;
        }
        // `close` takes `queues` before it sets `closed`, so the isolate is alive until the
        // guard is dropped.
        // SAFETY: the isolate is alive (see above), and `v8::Locker::IsLocked` may be called
        // from any thread.
        if unsafe { ffi::isolate_is_locked(self.isolate) } {
            drop(queues);
            self.reset(handles);
            return;
        }
        queues.deferred.push(handles);
    }

    /// Resets the handles queued by [`release`](Self::release). The deferred action hook calls
    /// it whenever `IsolateBase::Lock` locks the isolate.
    pub(crate) fn drain(&self, _isolate: &mut Isolate) {
        // The handles are reset outside the mutex, which only guards the queues themselves.
        let batch = std::mem::take(&mut self.lock_queues().deferred);
        for handles in batch {
            self.reset(handles);
        }
    }

    /// The number of handles waiting for the next lock.
    pub(crate) fn deferred_count(&self) -> usize {
        self.lock_queues().deferred.len()
    }

    /// Records that the isolate is about to be disposed, and resets the queued handles while
    /// it still can. Every handle of the isolate that is dropped from now on is left to
    /// `Dispose`.
    ///
    /// The `Isolate` reference proves that the caller holds the isolate lock; no handle of the
    /// isolate can be reset concurrently.
    pub(crate) fn close(&self, _isolate: &mut Isolate) {
        let batch = {
            let mut queues = self.lock_queues();
            self.closed.store(true, Ordering::Release);
            std::mem::take(&mut queues.deferred)
        };
        for handles in batch {
            self.reset(handles);
        }
    }

    /// Records that the isolate has been disposed, and frees the parked boxes, which V8 can no
    /// longer write into. Boxes released from now on are freed at once.
    pub(crate) fn dispose(&self) {
        let parked = {
            let mut queues = self.lock_queues();
            queues.disposed = true;
            std::mem::take(&mut queues.parked)
        };
        drop(parked);
    }

    /// Leaves the handles of a `Global` dropped after the state was closed to `Dispose`. A box
    /// stays allocated until then.
    fn park(&self, handles: Handles) {
        if let Handles::Boxed(slot) = handles {
            let mut queues = self.lock_queues();
            if !queues.disposed {
                queues.parked.push(slot);
            }
        }
    }

    /// Resets a dropped `Global`'s handles, in place: a box is freed only afterwards, once V8
    /// has forgotten its slot. The isolate must be locked by this thread.
    fn reset(&self, handles: Handles) {
        let mut handles = handles;
        match &mut handles {
            Handles::Inline(strong) => reset_strong(strong),
            Handles::Boxed(slot) => {
                reset_strong(&mut slot.strong);
                let traced = std::mem::replace(&mut slot.traced, TracedHandle::EMPTY);
                // SAFETY: the isolate is locked by this thread (guaranteed by caller), and
                // `traced` belongs to it.
                unsafe { traced.release(self.gc_epoch()) };
            }
        }
    }

    fn lock_queues(&self) -> MutexGuard<'_, Queues> {
        // The mutex is only held around `Vec` operations, so a poisoned lock still guards
        // consistent queues.
        self.queues.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Resets a strong handle, in place. The isolate must be locked by this thread.
#[expect(unsafe_code, reason = "resets a V8 handle")]
fn reset_strong(strong: &mut ffi::Global) {
    // SAFETY: the handle is valid and the isolate is locked (guaranteed by caller). `strong`
    // is not moved while pinned.
    unsafe { ffi::global_reset(Pin::new_unchecked(strong)) };
}

/// A `Global`'s traced handle, and the [GC epoch](IsolateState::gc_epoch) through which its
/// node is known to be live.
///
/// V8 frees traced nodes only while the epoch is odd, once the collection in progress has
/// finished marking: it frees the nodes that the marking did not reach. While the epoch is
/// even, an existing node lives at least until the next collection finishes marking. While it
/// is odd, a node that GC marks lives until the collection ends, and so does a node that GC
/// creates. workerd does not mark incrementally, so V8 does not allocate new nodes as marked;
/// but a node created during marking is marked by it: a tracing visit traces the node right
/// after it creates it (`GcVisitor::visit`), and a visit outside tracing that runs during
/// marking, when tracing makes a resource's last strong reference weak, is followed by a
/// tracing visit of the same resource (`Wrappable::visitRef`). A node created once marking is
/// over is created after V8 freed the nodes it did not reach. So a node is known to be live
/// through the even epoch at or after the moment GC last saw it in use. A node that GC saw in
/// use before the collection in progress is still there while the collection marks, so a GC
/// visit during marking may use it (see [`IsolateState::stale_before_visit`]); otherwise, a
/// node checked while the epoch is odd passes only if GC saw it in the same collection.
pub struct TracedHandle {
    /// Empty (`ptr == 0`) while the strong handle is strong.
    pub(crate) reference: ffi::TracedReference,
    /// The last epoch through which the node is known to be live. Meaningless while
    /// `reference` is empty.
    live_through: u64,
}

impl TracedHandle {
    pub(crate) const EMPTY: Self = Self {
        reference: ffi::TracedReference { ptr: 0 },
        live_through: 0,
    };

    /// Clears the reference without a reset if GC may have freed its node, because it was
    /// last known to be live before the epoch `stale_before`, and returns whether it did. The
    /// strong handle is then still weak.
    pub(crate) fn forget_if_stale(&mut self, stale_before: u64) -> bool {
        if self.reference.ptr == 0 || self.live_through >= stale_before {
            return false;
        }
        self.reference.ptr = 0;
        true
    }

    /// Records that GC visited the handle at `epoch`, which leaves a set reference live (see
    /// the type's documentation).
    pub(crate) fn note_visited(&mut self, epoch: u64) {
        if self.reference.ptr != 0 {
            self.live_through = epoch.next_multiple_of(2);
        }
    }

    /// Resets the reference if its node is known to be live at the current `epoch`, and leaves
    /// the node to V8 otherwise.
    ///
    /// # Safety
    ///
    /// The isolate the handle belongs to must be locked by this thread, and not closed.
    #[expect(unsafe_code, reason = "resets a V8 handle")]
    pub(crate) unsafe fn release(mut self, epoch: u64) {
        if self.forget_if_stale(epoch) || self.reference.ptr == 0 {
            return;
        }
        // SAFETY: the node is live, and the isolate is locked (guaranteed by caller).
        unsafe { ffi::traced_reference_reset(&mut self.reference) };
    }
}
