// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0

import { strictEqual, throws } from 'node:assert';

// Each of these values makes JSON.stringify() return undefined.
const notEncodable = {
  symbol: Symbol('x'),
  function: () => {},
  undefined: undefined,
  toJSONReturnsUndefined: {
    toJSON() {
      return undefined;
    },
  },
};

export const responseJsonNotEncodable = {
  async test(ctrl, env) {
    for (const [name, value] of Object.entries(notEncodable)) {
      if (env.pedantic) {
        throws(() => Response.json(value), TypeError, name);
      } else {
        const response = Response.json(value);
        strictEqual(await response.text(), 'undefined', name);
        strictEqual(response.headers.get('content-type'), 'application/json');
      }
    }
  },
};

export const responseJsonEncodable = {
  async test() {
    strictEqual(await Response.json(null).text(), 'null');
    strictEqual(await Response.json('undefined').text(), '"undefined"');
    strictEqual(await Response.json({ a: undefined }).text(), '{}');
    strictEqual(await Response.json([undefined]).text(), '[null]');
  },
};
