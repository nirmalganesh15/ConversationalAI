import { readSse } from './chat.store';

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
}

async function collect(body: ReadableStream<Uint8Array>) {
  const events = [];
  for await (const e of readSse(body)) events.push(e);
  return events;
}

describe('readSse', () => {
  it('parses named events with JSON payloads', async () => {
    const events = await collect(
      streamOf(
        'event: token\ndata: {"text":"Hel"}\n\n',
        'event: done\ndata: {"answer":"Hello"}\n\n',
      ),
    );
    expect(events).toEqual([
      { event: 'token', data: { text: 'Hel' } },
      { event: 'done', data: { answer: 'Hello' } },
    ]);
  });

  it('reassembles events split across network chunks and CRLF line endings', async () => {
    const events = await collect(streamOf('event: tok', 'en\r\ndata: {"te', 'xt":"a"}\r\n\r\n'));
    expect(events).toEqual([{ event: 'token', data: { text: 'a' } }]);
  });

  it('ignores keep-alive comments and non-JSON pings', async () => {
    const events = await collect(
      streamOf(': ping\n\n', 'data: not-json\n\n', 'event: x\ndata: {}\n\n'),
    );
    expect(events).toEqual([{ event: 'x', data: {} }]);
  });
});
