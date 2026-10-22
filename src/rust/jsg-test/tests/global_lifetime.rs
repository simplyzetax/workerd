// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

//! Tests for dropping a `Global` without the isolate lock, or after its isolate is gone.

use std::cell::Cell;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use jsg::ToJS;
use jsg::v8;
use jsg_macros::jsg_resource;

/// A resource whose only job is to own a `Global`, so that dropping the resource drops the
/// handle.
#[jsg_resource]
struct HandleHolder {
    handle: Cell<Option<v8::Global<v8::Value>>>,
}

#[jsg_resource]
impl HandleHolder {}

static HANDLE_HOLDER_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for HandleHolder {
    fn drop(&mut self) {
        HANDLE_HOLDER_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

/// A resource whose drops the tests count.
#[jsg_resource]
struct Counted;

#[jsg_resource]
impl Counted {}

static COUNTED_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for Counted {
    fn drop(&mut self) {
        COUNTED_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

/// A `Global` dropped while the isolate is not locked cannot reset its handle; the handle is
/// queued, keeps its value alive, and is reset the next time the isolate is locked.
#[test]
fn unlocked_drop_is_deferred_until_the_next_lock() {
    COUNTED_DROPS.store(0, Ordering::SeqCst);

    let harness = crate::Harness::new();
    let mut wrapper: Option<v8::Global<v8::Value>> = None;
    let mut baseline = 0;
    harness.run_in_context(|lock, _ctx| {
        // Created ahead of the measurements below, so that they only see the wrapper's handle.
        jsg::resource::function_template_of::<Counted>(lock);
        baseline = crate::Harness::used_global_handles_size(lock);

        let resource = jsg::Rc::new(Counted);
        wrapper = Some(resource.clone().to_js(lock).to_global(lock));
        // Only the `Global` keeps the wrapper, and with it the resource, alive now.
        drop(resource);
        Ok(())
    });
    let wrapper = wrapper.expect("created in the context");

    harness.run_in_context(|lock, _ctx| {
        crate::Harness::request_gc(lock);
        assert_eq!(
            COUNTED_DROPS.load(Ordering::SeqCst),
            0,
            "the Global roots the wrapper"
        );
        let with_wrapper = crate::Harness::used_global_handles_size(lock);
        assert!(with_wrapper > baseline);
        assert_eq!(lock.deferred_handle_count(), 0);

        crate::Harness::with_isolate_unlocked(lock, || drop(wrapper));
        assert_eq!(lock.deferred_handle_count(), 1);

        crate::Harness::request_gc(lock);
        assert_eq!(
            COUNTED_DROPS.load(Ordering::SeqCst),
            0,
            "the queued handle still roots the wrapper"
        );
        assert_eq!(crate::Harness::used_global_handles_size(lock), with_wrapper);
        Ok(())
    });

    // Locking the isolate again replays the queued release.
    harness.run_in_context(|lock, _ctx| {
        assert_eq!(lock.deferred_handle_count(), 0);
        crate::Harness::request_gc(lock);
        assert_eq!(COUNTED_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(crate::Harness::used_global_handles_size(lock), baseline);
        Ok(())
    });
}

/// A `Global` is `'static`, so it can outlive the harness, and with it the isolate. Dropping it
/// then must leave the handle alone: `v8::Isolate::Dispose` freed it with the isolate.
#[test]
fn global_dropped_after_its_isolate_is_disposed() {
    let harness = crate::Harness::new();
    let mut global: Option<v8::Global<v8::Value>> = None;
    harness.run_in_context(|lock, _ctx| {
        global = Some("escaped".to_js(lock).to_global(lock));
        Ok(())
    });
    drop(harness);
    drop(global.expect("created in the context"));
}

/// A resource that has a JavaScript wrapper and outlives its isolate is destroyed at once when
/// its last `Rc` goes, since the isolate detached its wrapper at teardown. The `Global` it owns
/// is dropped with it, after the isolate was disposed.
#[test]
fn wrapped_resource_dropped_after_its_isolate_is_disposed() {
    HANDLE_HOLDER_DROPS.store(0, Ordering::SeqCst);

    let harness = crate::Harness::new();
    let mut holder: Option<jsg::Rc<HandleHolder>> = None;
    harness.run_in_context(|lock, ctx| {
        let resource = jsg::Rc::new(HandleHolder {
            handle: Cell::new(None),
        });
        resource
            .handle
            .set(Some("held".to_js(lock).to_global(lock)));
        // Wrapping the resource ties its `Wrappable` to the isolate, which the isolate undoes
        // when it is torn down.
        ctx.set_global(lock, "holder", resource.clone());
        holder = Some(resource);
        Ok(())
    });
    drop(harness);
    assert_eq!(HANDLE_HOLDER_DROPS.load(Ordering::SeqCst), 0);

    drop(holder.expect("created in the context"));
    assert_eq!(HANDLE_HOLDER_DROPS.load(Ordering::SeqCst), 1);
}
