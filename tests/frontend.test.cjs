// Exercise the shipped inline functions with controllable responses, no CDN/browser dependency.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Syntax check the entire page, including initialization.

function load(context, name) {
  const pattern = new RegExp('(?:async )?function ' + name + '\\(');
  const start = script.search(pattern);
  assert.ok(start >= 0, name + ' exists');
  const end = script.indexOf('\n}', start) + 2;
  vm.runInContext(script.slice(start, end), context);
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness() {
  const requests = [], renders = [], errors = [];
  const context = vm.createContext({
    state: {},
    api(url) { const response = deferred(); requests.push({ url, ...response }); return response.promise; },
    renderResultsBody() { renders.push({ id: context.state.rsDetail?.id, report: context.state.rsReport?.id }); },
    renderExpDetail() { renders.push({ id: context.state.selectedExpDetail?.id }); },
    schedulePolling() {},
    toast(message) { errors.push(message); },
    document: { getElementById() { return { replaceChildren() { errors.push('detail error'); } }; } },
    emptyBox() {},
  });
  return { context, requests, renders, errors };
}

test('results keep the latest experiment when earlier report arrives late', async () => {
  const { context, requests, renders } = harness();
  load(context, 'selectResultsExp');
  const first = context.selectResultsExp('a');
  requests[0].resolve({ id: 'a' });
  await new Promise(setImmediate); // flush cross-context promise continuations
  const second = context.selectResultsExp('b');
  requests[2].resolve({ id: 'b' });
  await new Promise(setImmediate);
  requests[3].resolve({ id: 'b' });
  await second;
  requests[1].resolve({ id: 'a' });
  await first;
  assert.equal(context.state.rsDetail.id, 'b');
  assert.equal(context.state.rsReport.id, 'b');
  assert.deepEqual(renders.at(-1), { id: 'b', report: 'b' });
});

test('deselecting an experiment ignores its outstanding detail response', async () => {
  const { context, requests } = harness();
  load(context, 'selectResultsExp');
  const first = context.selectResultsExp('a');
  await context.selectResultsExp('');
  requests[0].resolve({ id: 'a' });
  await first;
  assert.equal(context.state.rsDetail, null);
  assert.equal(context.state.rsReport, null);
  assert.equal(requests.length, 1);
});

test('a repeated refresh of the same experiment rejects the older response', async () => {
  const { context, requests } = harness();
  load(context, 'refreshExpDetail');
  context.state.selectedExpId = 'a';
  const first = context.refreshExpDetail();
  const second = context.refreshExpDetail();
  requests[1].resolve({ id: 'a', status: 'plan_ready' });
  await second;
  requests[0].resolve({ id: 'a', status: 'planning' });
  await first;
  assert.equal(context.state.selectedExpDetail.status, 'plan_ready');
});

test('errors from a previous experiment cannot overwrite the new selection', async () => {
  const { context, requests, errors } = harness();
  load(context, 'refreshExpDetail');
  context.state.selectedExpId = 'a';
  const first = context.refreshExpDetail();
  context.state.selectedExpId = 'b';
  const second = context.refreshExpDetail();
  requests[1].resolve({ id: 'b' });
  await second;
  requests[0].reject(new Error('old network failure'));
  await first;
  assert.equal(context.state.selectedExpDetail.id, 'b');
  assert.deepEqual(errors, []);
});

function element(tag, attrs = {}, ...children) {
  const node = { tag, attrs: attrs || {}, children: children.flat().filter(x => x != null && x !== false),
    appendChild(child) { this.children.push(child); return child; }, addEventListener() {}, querySelectorAll() { return []; } };
  return node;
}

function walk(node, output = []) {
  if (node && typeof node === 'object') {
    output.push(node);
    for (const child of node.children || []) walk(child, output);
  }
  return output;
}

test('prototype names survive in object metrics and case IDs', () => {
  const context = vm.createContext({
    state: { caseFilter: 'all' }, h: element,
    clear(node) { node.children = []; }, pct: String, num: String, deltaTag: String,
    document: { createTextNode(text) { return { text }; } },
  });
  load(context, 'metricRow');
  load(context, 'renderReport');
  const names = ['__proto__', 'constructor', 'toString'];
  const side = {
    overall: { total: 3, correct: 3, accuracy: 1, latency: {}, per_object: names.map(object => ({ object, support: 1 })) },
    groups: {}, by_object_count: {},
    cases: names.map(id => ({ id, text: 'case ' + id, expected: [id], actual: [id], passed: true })),
  };
  const tree = context.renderReport({ id: 'run' }, {
    baseline: side, candidate: side, accuracy_delta: 0, improved_ids: [], regressed_ids: [],
    training_verified: false, source: 'user_imported_predictions',
  });
  const rows = walk(tree).filter(node => node.tag === 'tr');
  const metricRows = rows.filter(row => row.children.length === 8 && row.children[0]?.tag === 'td');
  assert.deepEqual(metricRows.map(row => row.children[0].attrs.text), names.slice().sort());
  const caseRows = rows.filter(row => row.children.length === 6 && row.children[0]?.tag === 'td');
  assert.deepEqual(caseRows.map(row => row.children[0].attrs.text), names);
});
