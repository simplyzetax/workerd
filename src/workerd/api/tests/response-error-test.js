// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0
import { strictEqual, throws } from 'node:assert';
import { env } from 'cloudflare:workers';

// Response.error() must work outside of an IoContext.
const globalScopeResponse = Response.error();

function check(response) {
  strictEqual(response.type, 'error');
  strictEqual(response.status, 0);
  if (env.pedantic) {
    throws(() => response.headers.append('foo', 'bar'), {
      name: 'TypeError',
      message: "Can't modify immutable headers.",
    });
    throws(() => response.headers.set('foo', 'bar'), { name: 'TypeError' });
    throws(() => response.headers.delete('foo'), { name: 'TypeError' });
    strictEqual(response.headers.get('foo'), null);
  } else {
    response.headers.append('foo', 'bar');
    response.headers.set('baz', 'qux');
    strictEqual(response.headers.get('foo'), 'bar');
    strictEqual(response.headers.get('baz'), 'qux');
  }
}

export const errorHeaders = {
  test() {
    check(Response.error());
  },
};

export const errorHeadersClone = {
  test() {
    check(Response.error().clone());
  },
};

export const errorHeadersAtGlobalScope = {
  test() {
    check(globalScopeResponse);
  },
};
