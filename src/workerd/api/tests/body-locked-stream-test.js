// Copyright (c) 2026 Cloudflare, Inc.
// Licensed under the Apache 2.0 license found in the LICENSE file or at:
//     https://opensource.org/licenses/Apache-2.0
import { doesNotThrow, throws } from 'assert';

// WPT covers new Response(lockedStream) under pedantic_wpt. Its Request counterpart passes
// regardless, because the relative URL it uses is rejected first. WPT also always runs with
// pedantic_wpt, so the default behavior is covered here.

function lockedStream() {
  const stream = new ReadableStream();
  stream.getReader();
  return stream;
}

export const lockedStreamBody = {
  test(_, env) {
    const makeRequest = () =>
      new Request('https://example.com', {
        method: 'POST',
        body: lockedStream(),
      });
    const makeResponse = () => new Response(lockedStream());
    if (env.pedantic) {
      throws(makeRequest, {
        name: 'TypeError',
        message: /locked to a reader/,
      });
    } else {
      doesNotThrow(makeRequest);
      doesNotThrow(makeResponse);
    }
  },
};
