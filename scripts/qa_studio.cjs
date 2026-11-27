/* Browser QA in real Chromium, native or GitHub Pages build. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path');
const {execFileSync}=require('node:child_process');
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8765',browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const done=async stage=>{
   await page.waitForFunction(s=>document.querySelector('#stage').textContent===s||document.querySelector('#dot').classList.contains('error'),stage,{timeout:300000});
   assert.equal(await page.locator('#stage').textContent(),stage,await page.locator('#error').textContent());
  };
  await page.goto(base); await page.locator('#sample').click();
  await page.locator('#cut').selectOption('strips');await page.locator('#count').fill('257');await page.locator('#maxSide').selectOption('128');
  await page.locator('#run').click();await done('Fragments generated');await page.locator('#transfer').waitFor({state:'visible'});
  assert.equal(await page.locator('#genCount').textContent(),'257 / 257');
  assert.match(await page.locator('#fitInfo').textContent(),/Auto-fitted/);
  await page.locator('#count').fill('16');await page.locator('#cut').selectOption('jigsaw');await page.locator('#maxSide').selectOption('2048');
  await page.locator('#run').click();await done('Fragments generated');await page.locator('#transfer').waitFor({state:'visible'});
  assert.equal(await page.locator('#genCount').textContent(),'16 / 16');
  assert.equal(await page.locator('#reconstructionOutput').isHidden(),true);
  const pieces=path.join('/tmp','shardelix-v13-replay.zip'),truth=path.join('/tmp','shardelix-v13-truth.zip'),source=path.join('/tmp','shardelix-v13-source.png');
  for(const [id,file] of [['downloadFragments',pieces],['downloadTruth',truth],['downloadSource',source]]){
   const [d]=await Promise.all([page.waitForEvent('download'),page.locator('#'+id).click()]);await d.saveAs(file);
  }
  if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-generator.png'),fullPage:true});
  await page.locator('#transfer').click();
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('"pieces":16'),{},{timeout:60000});
  assert.equal(await page.locator('#truthLabel').textContent(),'ground_truth.zip');
  await page.locator('#run').click();await done('Reconstruction exported');
  await page.locator('#resultImg').waitFor({state:'visible'});assert.equal(await page.locator('#placed').textContent(),'16 / 16');assert.equal(await page.locator('#dpa').textContent(),'100.00%');
  assert.notEqual(await page.locator('#rawDpa').textContent(),'—');
  if(await page.locator('#alignedWrap').isVisible()){
   assert.match(await page.locator('#orientationLabel').textContent(),/Reference-aligned/);
   await page.locator('#alignedPreview').uncheck();
   await page.waitForFunction(()=>document.querySelector('#orientationLabel').textContent==='Blind reconstruction');
   await page.locator('#alignedPreview').check();
  }
  if(process.env.EXPECT_CPP)assert.equal(await page.locator('#engineBadge').textContent(),process.env.EXPECT_CPP);
  if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT,fullPage:true});
  await page.locator('#clearTruth').click();await page.locator('#upload').setInputFiles(pieces);
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('"pieces":16'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'—');assert.equal(await page.locator('#rawDpa').textContent(),'—');assert.equal(await page.locator('#alignedWrap').isHidden(),true);
  await page.locator('#truthUpload').setInputFiles(truth);await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'100.00%');
  await page.locator('#truthUpload').setInputFiles(source);await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'—');
  await page.locator('#comparisonInfo').waitFor({state:'visible'});assert.match(await page.locator('#comparisonInfo').textContent(),/MAE/);
  await page.locator('#imageTab').click();
  assert.equal(await page.locator('#genCount').textContent(),'16 / 16');
  await page.locator('#batch').check();await page.locator('#counts').fill('16, 25');await page.locator('#seeds').fill('42');await page.locator('#profile').selectOption('balanced');await page.locator('#run').click();await done('Benchmark complete');assert.equal(await page.locator('#benchRows tr').count(),2);
  // External equal-sized RGB/RGBA tiles, no case.json: infer dimensions.
  await page.locator('#batch').uncheck();await page.locator('#cut').selectOption('grid');
  await page.locator('#angle').selectOption('none');await page.locator('#count').fill('16');await page.locator('#maxSide').selectOption('128');
  await page.locator('#run').click();await done('Fragments generated');await page.locator('#transfer').waitFor({state:'visible'});
  const external=path.join('/tmp','shardelix-no-metadata.zip'),generated=path.join('/tmp','shardelix-grid-with-metadata.zip');
  const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#downloadFragments').click()]);await download.saveAs(generated);
  execFileSync(process.env.PYTHON||'python',['-c','import sys,zipfile; a=zipfile.ZipFile(sys.argv[1]); b=zipfile.ZipFile(sys.argv[2],"w"); [b.writestr(n,a.read(n)) for n in a.namelist() if n.startswith("pieces/")]; b.close(); a.close()',generated,external]);
  await page.locator('#fragmentTab').click();if(await page.locator('#clearTruth').isVisible())await page.locator('#clearTruth').click();await page.locator('#upload').setInputFiles(external);
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('16 opaque tiles'),{},{timeout:30000});
  assert.equal(await page.locator('#rows').inputValue(),'4');assert.equal(await page.locator('#cols').inputValue(),'4');
  assert.equal(await page.locator('#width').inputValue(),'128');assert.equal(await page.locator('#height').inputValue(),'128');
  assert.equal(await page.locator('#useMeta').isChecked(),true);assert.equal(await page.locator('#geometryFields').isHidden(),true);
  // Deliberately corrupt stale manual inputs. Automatic mode must ignore these
  // at preflight AND solve time; the user need only click Run.
  await page.evaluate(()=>{for(const id of ['rows','cols','width','height'])document.querySelector('#'+id).value='1';
     document.querySelector('#rows').dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal(await page.locator('#run').isEnabled(),true);
  await page.locator('#run').click();await done('Reconstruction exported');await page.locator('#resultImg').waitFor({state:'visible'});assert.equal(await page.locator('#placed').textContent(),'16 / 16');
  assert.match(await page.locator('#diagnostics').textContent(),/inferred-grid/);
  await page.locator('#useMeta').uncheck();
  await page.locator('#inferredLayout').selectOption({label:'2 × 8 · 256 × 64 px'});assert.equal(await page.locator('#width').inputValue(),'256');
  await page.locator('#inferredLayout').selectOption({label:'4 × 4 · 128 × 128 px'});assert.equal(await page.locator('#width').inputValue(),'128');
  await page.locator('#useMeta').check();
  // Scan mode: a single opaque photo containing multiple disconnected pieces,
  // with no metadata or canvas. Auto mode must route to paper segmentation.
  await page.locator('#inputMode').selectOption('auto');
  const scanData=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=300;c.height=180;
    const ctx=c.getContext('2d');ctx.fillStyle='#202020';ctx.fillRect(0,0,300,180);
    ctx.fillStyle='#b8e0cf';ctx.fillRect(20,30,80,80);ctx.fillStyle='#dd9887';ctx.fillRect(180,50,75,100);
    return c.toDataURL('image/png').split(',')[1];});
  await page.locator('#upload').setInputFiles({name:'scan-overview.png',mimeType:'image/png',buffer:Buffer.from(scanData,'base64')});
  await page.waitForFunction(()=>document.querySelector('#scanHelp').hidden===false,{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');
  await page.locator('#scanPreview').waitFor({state:'visible'});
  assert.match(await page.locator('#notice').textContent(),/2 detected from 1 images/);
  assert.equal(await page.locator('#dpa').textContent(),'—');
  assert.match(await page.locator('#diagnostics').textContent(),/physical-scan-se2-experimental/);
  assert.ok(await page.getByRole('button',{name:'Scan processing report'}).isVisible());
  for(const width of [1024,768,390,320]){
   await page.setViewportSize({width,height:844});
   const mobile=await page.evaluate(()=>({viewport:window.innerWidth,scroll:document.documentElement.scrollWidth,overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>window.innerWidth+1}).slice(0,20).map(e=>[e.tagName,e.id,String(e.className),Math.round(e.getBoundingClientRect().right)])}));
   assert.ok(mobile.scroll<=mobile.viewport,JSON.stringify(mobile));
   if(width===390&&process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-mobile.png'),fullPage:true});
  }
  assert.deepEqual(errors,[]);console.log("PASS: generator, transfer, separate truth, blind ZIP replay, image reference, batch, 320–1440 responsive, no JS errors");
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
