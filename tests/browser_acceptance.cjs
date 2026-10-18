// Optional real-browser acceptance. Requires Playwright and an installed Chromium.
// All data, credentials and processes are disposable fixtures; never calls a model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const { spawn, spawnSync } = require('node:child_process');
const { chromium } = require('playwright');

async function main() {
  const questionCount = Number(process.env.KEV_BROWSER_QUESTION_COUNT || 50);
  const businessScope = process.env.KEV_BROWSER_BUSINESS_SCOPE || '';
  const explicitQuestions = process.env.KEV_BROWSER_LABEL_MODE === 'explicit_questions';
  assert.ok(Number.isInteger(questionCount) && questionCount >= 1 && questionCount <= 100);
  const intents = row => row.intents || Object.keys(row.questions).filter(key => row.questions[key].label);
  const root = path.resolve(__dirname, '..');
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'kev-browser-'));
  const project = path.join(temporary, 'project'); fs.mkdirSync(project);
  for (const name of fs.readdirSync(root).filter(name => name.endsWith('.py') || ['models.json', 'kev-source-contract.json'].includes(name))) {
    fs.copyFileSync(path.join(root, name), path.join(project, name));
  }
  for (const name of ['templates', 'examples', 'scripts']) fs.cpSync(path.join(root, name), path.join(project, name), { recursive: true });
  fs.symlinkSync(path.join(root, '.venv'), path.join(project, '.venv'), 'dir');
  const output = process.env.KEV_BROWSER_OUTPUT || path.join(temporary, 'evidence'); fs.mkdirSync(output, { recursive: true });
  const port = await new Promise(resolve => { const server = net.createServer(); server.listen(0, '127.0.0.1', () => { const value = server.address().port; server.close(() => resolve(value)); }); });
  const base = 'http://127.0.0.1:' + port;
  const inherited = Object.fromEntries(['PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','TMPDIR','SYSTEMROOT'].filter(key => process.env[key]).map(key => [key,process.env[key]]));
  const environment = { ...inherited, KEV_TRAINER_DATA: path.join(temporary, 'state'), CLAUDE_BIN: path.join(temporary, 'absent-cli'),
    KEV_WORKER_TOKEN: 'browser-only-ephemeral-fixture-' + 'x'.repeat(32) };
  const python = path.join(project, '.venv/bin/python');
  const log = fs.openSync(path.join(temporary, 'service.log'), 'w');
  const server = spawn(python, [path.join(project, 'server.py'), '--port', String(port)], { cwd: project, env: environment, stdio: ['ignore', log, log] });
  let browser;
  const errors = [];
  try {
    for (let i = 0; i < 100; i++) {
      try { const response = await fetch(base + '/api/health'); if (response.ok) break; } catch {}
      if (server.exitCode !== null) throw new Error(fs.readFileSync(path.join(temporary, 'service.log'), 'utf8'));
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    browser = await chromium.launch({ executablePath: process.env.KEV_BROWSER_EXECUTABLE || '/usr/bin/chromium', headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    assert.match(await page.locator('#health-content').innerText(), /未接入/);
    await page.locator('#ex-load-btn').click();
    await page.locator('#ds-list').getByRole('button', { name: '预览', exact: true }).first().waitFor();
    await page.locator('#ds-list').getByRole('button', { name: '预览', exact: true }).first().click();
    await page.locator('#modal-back.open').waitFor();
    await page.locator('#modal-close').click();

    const objects = Array.from({ length: questionCount }, (_, i) => ({ id: 'object-' + String(i).padStart(2, '0'),
      instructions: 'Synthetic browser object ' + i, criteria: { true: 'fixture yes', false: 'fixture no' } }));
    if (businessScope) await page.locator('#task-scope').fill(businessScope);
    if (explicitQuestions) await page.locator('#task-label-mode').selectOption('explicit_questions');
    await page.locator('#task-file').setInputFiles({ name: 'fixture-task.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ name: '浏览器合成任务（非真实目录）', objects })) });
    await page.locator('#task-upload').click();
    await page.waitForFunction(() => document.getElementById('up-task').value !== '');
    const versions = {}, rows = {};
    for (const role of ['train', 'calibration', 'validation', 'test', 'regression']) {
      rows[role] = [0, 1].map(i => {
        const row = { id: 'browser-' + role + i, text: 'browser fixture ' + role + i,
          intents: i === 0 ? ['object-00'] : [], group_id: role + '-group-' + i };
        if (explicitQuestions) {
          row.questions = { [objects[i === 0 ? 0 : objects.length-1].id]: { type: 'noul', label: i === 0 } };
          delete row.intents;
        }
        return row;
      });
      await page.locator('#up-role').selectOption(role);
      await page.locator('#up-source').fill('Synthetic browser fixture; no real training data');
      await page.locator('#up-file').setInputFiles({ name: role + '-fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(rows[role])) });
      const response = page.waitForResponse(response => response.url().endsWith('/api/datasets') && response.request().method() === 'POST');
      await page.locator('#up-btn').click(); versions[role] = await (await response).json();
      await page.waitForFunction(() => !document.getElementById('up-btn').disabled);
    }
    await page.locator('#nav [data-view=create]').click();
    await page.locator('#manual-name').fill('浏览器手动验收（仅夹具）');
    await page.locator('#manual-count').fill('1');
    await page.locator('#manual-minutes').fill('5');
    for (const role of Object.keys(versions)) await page.locator('#manual-' + role).selectOption(versions[role].id);
    const created = page.waitForResponse(response => response.url().endsWith('/api/manual-experiments'));
    await page.locator('#manual-create').click(); const experiment = await (await created).json();
    await page.locator('#exp-detail form [name=learning_rate]').waitFor();
    assert.equal(await page.getByRole('button',{name:'只测原模型基线（不创建 Trial）',exact:true}).isDisabled(),true);
    for (const [name, value] of Object.entries({ learning_rate: '.0001', epochs: '1', batch_size: '1', gradient_accumulation: '1', seed: '42' })) {
      await page.locator('#exp-detail form [name=' + name + ']').fill(value);
    }
    await page.locator('#exp-detail form [name=precision]').selectOption('fp32');
    const saved = page.waitForResponse(response => response.url().endsWith('/trials') && response.request().method() === 'POST');
    await page.getByRole('button', { name: '保存 Trial', exact: true }).click(); const trial = await (await saved).json();
    await page.getByRole('button', { name: '提交 CPU 准备检查', exact: true }).click();
    await page.locator('[data-job-state=QUEUED]').waitFor();
    const worker = spawnSync(python, [path.join(project, 'worker.py'), '--server', base, '--once'], { cwd: project, env: environment, encoding: 'utf8', timeout: 15000 });
    assert.equal(worker.status, 0, worker.stdout + worker.stderr);
    const result = JSON.parse(worker.stdout.trim().split('\n').at(-1)); assert.equal(result.training_verified, false); assert.equal(result.state, 'SUCCEEDED');
    await page.getByRole('button', { name: '刷新准备检查状态', exact: true }).click();
    await page.locator('[data-job-state=SUCCEEDED]').waitFor();
    const validation = { dataset_sha256: versions.validation.sha256, predictions: rows.validation.map(row => ({ id: row.id, intents: intents(row) })) };
    await page.locator('#exp-detail input[type=file]').setInputFiles({ name: 'validation-fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(validation)) });
    await page.getByRole('button', { name: '导入 validation 预测', exact: true }).click();
    await page.getByRole('button', { name: '按 validation 选定此候选', exact: true }).click();
    await page.getByRole('button', { name: '已选候选', exact: true }).waitFor();
    if (businessScope) await page.getByText('按业务场景验收', { exact: true }).waitFor();
    if (explicitQuestions) assert.match(await page.locator('#exp-detail').innerText(), /未询问问题/);
    await page.waitForFunction(() => document.getElementById('toasts').children.length === 0);
    await page.screenshot({ path: path.join(output, 'manual-desktop.png'), fullPage: true });

    // Separate disposable experiment: review/start/cancel a proposal while GPU is disabled.
    await page.locator('#nav [data-view=create]').click();
    await page.locator('#manual-name').fill('浏览器方案审查（仅夹具）');
    for (const role of Object.keys(versions)) await page.locator('#manual-' + role).selectOption(versions[role].id);
    const optimizationCreated = page.waitForResponse(response => response.url().endsWith('/api/manual-experiments'));
    await page.locator('#manual-create').click(); await optimizationCreated;
    await page.getByRole('button',{name:'新建方案审查（0 额度可离线导入）',exact:true}).click();
    await page.getByRole('button',{name:'导入结构化方案 JSON',exact:true}).waitFor();
    const plan = {schema_version:2,objective:'合成方案审查，非模型效果',hypotheses:['只验证软件审批流程'],
      trials:[{id:'fixture-1',parameters:{learning_rate:.0001,epochs:1,batch_size:1,gradient_accumulation:1,seed:43,precision:'fp32'},rationale:'合成候选'}],
      stop_conditions:{metric:'validation_accuracy',target_accuracy:1,min_improvement:0,patience:1,max_baseline_regression:.1}};
    await page.locator('#exp-detail input[type=file]').setInputFiles({name:'plan.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(plan))});
    await page.getByRole('button',{name:'导入结构化方案 JSON',exact:true}).click();
    await page.getByLabel('审查人',{exact:true}).fill('浏览器夹具审查人');
    await page.getByLabel('审查意见',{exact:true}).fill('已审核当前 SHA、参数和预算；未批准真实训练或收费调用');
    await page.getByRole('button',{name:'批准当前 SHA 方案',exact:true}).click();
    await page.getByRole('button',{name:'开始已批准的串行候选',exact:true}).click();
    await page.getByRole('heading',{name:/WAITING_RESOURCE/}).waitFor();
    await page.waitForFunction(() => document.getElementById('toasts').children.length === 0);
    await page.screenshot({path:path.join(output,'optimization-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true,'optimization mobile overflow');
    await page.screenshot({path:path.join(output,'optimization-mobile.png'),fullPage:true});
    await page.getByRole('button',{name:'取消自动方案',exact:true}).click();
    await page.getByRole('heading',{name:/CANCELLED/}).waitFor();
    await page.setViewportSize({width:1440,height:1000});
    await page.locator('#exp-list .exp-item').filter({hasText:'浏览器手动验收（仅夹具）'}).click();
    await page.getByRole('button',{name:'已选候选',exact:true}).waitFor();

    await page.locator('#nav [data-view=results]').click();
    await page.locator('#rs-exp').selectOption(experiment.id);
    await page.locator('#rs-body input[type=file]').first().waitFor();
    for (const [index, side] of ['baseline', 'candidate'].entries()) {
      const payload = { dataset_sha256: versions.test.sha256, trial_id: trial.id, model: 'browser-fixture-' + side,
        predictions: rows.test.map(row => ({ id: row.id, intents: side === 'candidate' ? intents(row) : [],
          probabilities: Object.fromEntries((row.questions ? Object.keys(row.questions) : objects.map(obj => obj.id))
            .map(key => [key, side === 'candidate' && intents(row).includes(key) ? 1 : 0])) })) };
      const input = page.locator('#rs-body input[type=file]').nth(index);
      await input.setInputFiles({ name: side + '-fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(payload)) });
      const response = page.waitForResponse(response => response.url().endsWith('/predictions/' + side));
      await input.locator('..').getByRole('button', { name: '导入并评测', exact: true }).click();
      assert.equal((await response).status(), 200);
      await page.waitForFunction(() => Array.from(document.querySelectorAll('#rs-body input[type=file]')).every(input => input.files.length === 0));
    }
    await page.getByRole('heading', { name: '评测报告', exact: true }).waitFor();
    assert.match(await page.locator('#rs-body').innerText(), /training_verified: false/);
    assert.match(await page.locator('#rs-body').innerText(), /配对收益不确定/);
    if (businessScope) await page.locator('#rs-body').getByText('按业务场景验收', { exact: true }).waitFor();
    if (explicitQuestions) assert.match(await page.locator('#rs-body').innerText(), /未询问的问题不作为负例/);
    const downloadEvent = page.waitForEvent('download');
    await page.getByRole('link', { name: '下载 report.json', exact: true }).click();
    const download = await downloadEvent; await download.saveAs(path.join(output, 'fixture-report.json'));
    assert.equal(JSON.parse(fs.readFileSync(path.join(output, 'fixture-report.json'))).training_verified, false);
    await page.waitForFunction(() => document.getElementById('toasts').children.length === 0);
    await page.screenshot({ path: path.join(output, 'results-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    for (const view of ['data', 'create', 'experiments', 'results']) {
      await page.locator('#nav [data-view=' + view + ']').click();
      await page.waitForTimeout(150);
      const fits = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
      if (!fits) {
        await page.screenshot({ path: path.join(output, 'overflow-' + view + '.png'), fullPage: true });
        console.error(await page.evaluate(() => Array.from(document.querySelectorAll('body *')).filter(el => {
          const box = el.getBoundingClientRect(); return box.width > 0 && box.right > innerWidth + 1;
        }).slice(0, 20).map(el => ({ tag: el.tagName, id: el.id, className: el.className,
          right: el.getBoundingClientRect().right, text: el.textContent.slice(0, 70) }))));
      }
      assert.equal(fits, true, 'horizontal overflow in ' + view);
    }
    await page.screenshot({ path: path.join(output, 'results-mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ browser: await browser.version(), javascript_errors: errors, widths_checked: [1440, 390],
      question_count: questionCount, business_scope: businessScope, explicit_questions: explicitQuestions,
      manual_preparation: 'passed', validation_before_test: 'passed', proposal_review_disabled_gpu_start_cancel:'passed', report_download: 'passed', training_verified: false, evidence: output }));
  } finally {
    if (browser) await browser.close();
    const stopped = spawnSync(python, [path.join(project, 'server.py'), '--stop'], { cwd: project, env: environment, encoding: 'utf8', timeout: 20000 });
    if (server.exitCode === null) { server.kill('SIGTERM'); await new Promise(resolve => server.once('exit', resolve)); }
    fs.closeSync(log);
    if (stopped.status !== 0) console.error('Stop evidence:', stopped.stdout, stopped.stderr);
    if (process.env.KEV_BROWSER_OUTPUT) fs.rmSync(temporary, { recursive: true, force: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
