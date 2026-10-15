// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

// A Durable Object keeps a single IoContext for its whole lifetime, but gets a separate
// IncomingRequest for each event delivered to it. These tests cover promises that an actor event
// creates and that some other IoContext settles later, which routes the settlement back through the
// actor's delete queue.
//
// Settling such a promise runs application JavaScript, which needs a current IncomingRequest. The
// continuations below use timers and storage, which both depend on one.

import { DurableObject } from 'cloudflare:workers';
import { deepStrictEqual, strictEqual } from 'node:assert';

// Polls until `read()` returns something. Used instead of a fixed sleep wherever a test is waiting
// for progress rather than for the absence of it.
async function waitFor(what, read) {
  for (let i = 0; i < 2000; i++) {
    const value = read();
    if (value !== undefined) return value;
    await scheduler.wait(1);
  }
  throw new Error(`timed out waiting for ${what}`);
}

// Exercises request-scoped services: scheduler.wait() schedules against the request's timer, and
// storage goes through the actor's cache.
async function useRequestScopedApis(ctx, key, value) {
  await scheduler.wait(1);
  await ctx.storage.put(key, value);
  return { value, stored: await ctx.storage.get(key) };
}

export class TestActor extends DurableObject {
  // Holds its event open by awaiting the promise, so this request is still current when the
  // settlement arrives.
  async awaitFromActiveRequest() {
    const { promise, resolve } = Promise.withResolvers();
    globalThis.activeResolve = resolve;
    const value = await promise;
    return await useRequestScopedApis(this.ctx, 'active', value);
  }

  // Responds immediately but retains the continuation with ctx.waitUntil(), so the request outlives
  // the response and is still current when the settlement arrives.
  async createRetainedPromise() {
    const { promise, resolve } = Promise.withResolvers();
    globalThis.retainedResolve = resolve;
    this.ctx.waitUntil(
      promise.then(async (value) => {
        globalThis.retainedResult = await useRequestScopedApis(
          this.ctx,
          'retained',
          value
        );
      })
    );
    return 'created';
  }

  // Responds without retaining the continuation, so this event's IncomingRequest drains as soon as
  // the RPC session ends and the actor is left with no current request.
  async createDetachedPromise() {
    const { promise, resolve } = Promise.withResolvers();
    globalThis.detachedResolve = resolve;
    this.detached = promise.then(async (value) => {
      // Recorded before the first await, so a test can distinguish "the reaction started" from "the
      // reaction finished".
      globalThis.detachedReactionRan = true;
      return await useRequestScopedApis(this.ctx, 'detached', value);
    });
    return 'created';
  }

  async getDetachedResult() {
    return await this.detached;
  }

  // Like createDetachedPromise(), but the reaction only records the value, so a test can tell that
  // it ran without awaiting it from an event.
  createUnobservedPromise() {
    const { promise, resolve } = Promise.withResolvers();
    globalThis.unobservedResolve = resolve;
    promise.then((value) => {
      globalThis.unobservedValue = value;
    });
    return 'created';
  }

  // An event that ends at once, without awaiting anything.
  ping() {
    return 'pong';
  }

  // Keeps a signal received over RPC without observing it, so its RPC subscription stays unarmed.
  holdSignal(signal) {
    globalThis.heldSignal = signal;
    return 'held';
  }

  // Observes the held signal from this actor, a context other than the one that received it, and
  // stops observing it again, `times` times over. Returns the number of actions then queued on the
  // receiver.
  churnHeldSignalListener(times) {
    const signal = globalThis.heldSignal;
    const listener = () => {};
    for (let i = 0; i < times; i++) {
      signal.addEventListener('abort', listener);
      signal.removeEventListener('abort', listener);
    }
    return signal.getRpcReceiverQueuedActionCountForTest();
  }

  // Publishes a resolver from inside blockConcurrencyWhile(), then keeps the critical section, and
  // with it the input lock, held for a while after the test has settled the promise.
  async settleDuringCriticalSection() {
    const order = [];
    const { promise, resolve } = Promise.withResolvers();
    const settled = promise.then(() => order.push('settled'));
    await this.ctx.blockConcurrencyWhile(async () => {
      globalThis.criticalResolve = resolve;
      while (!globalThis.criticalResolved) {
        await scheduler.wait(1);
      }
      await scheduler.wait(10);
      order.push('critical section done');
    });
    await settled;
    return order;
  }
}

export const settlementRunsUnderAnActiveRequest = {
  async test(_, env) {
    const stub = env.ns.get(env.ns.idFromName('active'));
    const pending = stub.awaitFromActiveRequest();

    // Publishing the resolver is the actor's last act before parking on the await.
    const resolve = await waitFor(
      'the actor to publish its resolver',
      () => globalThis.activeResolve
    );
    globalThis.activeResolve = undefined;
    resolve('active-value');

    const result = await pending;
    strictEqual(result.value, 'active-value');
    strictEqual(result.stored, 'active-value');
  },
};

export const settlementRunsUnderADrainingRequest = {
  async test(_, env) {
    const stub = env.ns.get(env.ns.idFromName('retained'));
    strictEqual(await stub.createRetainedPromise(), 'created');

    const resolve = globalThis.retainedResolve;
    globalThis.retainedResolve = undefined;
    resolve('retained-value');

    // The actor receives no further event, so the settlement runs while the request drains, kept
    // alive by its waitUntil task.
    const result = await waitFor(
      'the retained continuation to run',
      () => globalThis.retainedResult
    );
    strictEqual(result.value, 'retained-value');
    strictEqual(result.stored, 'retained-value');
  },
};

export const settlementWaitsForTheNextEventWhenTheActorIsIdle = {
  async test(_, env) {
    const stub = env.ns.get(env.ns.idFromName('detached'));
    strictEqual(await stub.createDetachedPromise(), 'created');

    // Nothing retains that event's request, so it drains once the RPC session ends. Give it time to
    // do so before settling the promise.
    await scheduler.wait(100);

    globalThis.detachedResolve('detached-value');
    globalThis.detachedResolve = undefined;

    // With no request to run under, the settlement stays queued.
    await scheduler.wait(100);
    strictEqual(globalThis.detachedReactionRan, undefined);

    // The actor is still healthy, and its next event picks the settlement up.
    const result = await stub.getDetachedResult();
    strictEqual(globalThis.detachedReactionRan, true);
    strictEqual(result.value, 'detached-value');
    strictEqual(result.stored, 'detached-value');
  },
};

export const settlementRunsAsPartOfAShortNextEvent = {
  async test(_, env) {
    const stub = env.ns.get(env.ns.idFromName('short'));
    strictEqual(await stub.createUnobservedPromise(), 'created');

    // As in the previous test, settle the promise once the actor is idle.
    await scheduler.wait(100);
    globalThis.unobservedResolve('unobserved-value');
    globalThis.unobservedResolve = undefined;
    await scheduler.wait(100);
    strictEqual(globalThis.unobservedValue, undefined);

    // The next event ends at once, without awaiting the settlement, but its request still runs it
    // before finishing. Nothing would run it afterwards.
    strictEqual(await stub.ping(), 'pong');
    strictEqual(
      await waitFor('the settlement to run', () => globalThis.unobservedValue),
      'unobserved-value'
    );
  },
};

export const anIdleActorQueuesOneSubscriptionRequestPerSignal = {
  async test(_, env) {
    const receiver = env.ns.get(env.ns.idFromName('signal-receiver'));
    const producer = env.ns.get(env.ns.idFromName('signal-producer'));
    const controller = new AbortController();
    strictEqual(await receiver.holdSignal(controller.signal), 'held');
    const signal = globalThis.heldSignal;

    // The capability that lets this context abort the signal keeps the receiver's RPC session, and
    // with it the receiver's request, open. Aborting releases it, ending the session. The receiver
    // doesn't observe the signal, so the abort only reaches its pending reason: observers
    // registered later still need the receiver to arm the subscription.
    controller.abort('done');
    await waitFor('the abort to reach the receiver', () =>
      signal.aborted ? true : undefined
    );

    // Give the receiver's request time to finish draining. Each observer registered from a context
    // other than the receiver's asks the receiver to arm the subscription. That request then stays
    // queued while the event loop idles, which it only does if the receiver has no request to run
    // it under.
    await scheduler.wait(100);
    strictEqual(await producer.churnHeldSignalListener(100), 1);
    await scheduler.wait(100);
    strictEqual(signal.getRpcReceiverQueuedActionCountForTest(), 1);

    // The producer's later requests and registrations don't add to it.
    for (let i = 0; i < 4; i++) {
      strictEqual(await producer.churnHeldSignalListener(100), 1);
    }

    // The receiver's next event arms the subscription, which delivers the abort to the signal's
    // observers.
    const aborted = new Promise((resolve) => {
      signal.addEventListener('abort', () => resolve(signal.reason));
    });
    strictEqual(await receiver.ping(), 'pong');
    strictEqual(await aborted, 'done');
    strictEqual(signal.getRpcReceiverQueuedActionCountForTest(), 0);
  },
};

export const settlementWaitsForACriticalSectionToEnd = {
  async test(_, env) {
    const pending = env.ns
      .get(env.ns.idFromName('critical'))
      .settleDuringCriticalSection();
    const resolve = await waitFor(
      'the critical section to start',
      () => globalThis.criticalResolve
    );
    globalThis.criticalResolve = undefined;
    resolve();
    globalThis.criticalResolved = true;
    deepStrictEqual(await pending, ['critical section done', 'settled']);
  },
};
