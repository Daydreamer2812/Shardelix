/* Browser QA in real Chromium, native or GitHub Pages build. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),path=require('node:path');
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8765',browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const done=async stage=>{
   await page.waitForFunction(s=>document.querySelector('#stage').textContent===s||document.querySelector('#dot').classList.contains('error'),stage,{timeout:300000});
   assert.equal(await page.locator('#stage').textContent(),stage,await page.locator('#error').textContent());
  };
  await page.goto(base); await page.locator('#sample').click(); await page.locator('#count').fill('16');await page.locator('#cut').selectOption('jigsaw');
  await page.locator('#run').click();await done('Fragments generated');await page.locator('#transfer').waitFor({state:'visible'});
  assert.equal(await page.locator('#genCount').textContent(),'16 / 16');
  assert.equal(await page.locator('#reconstructionOutput').isHidden(),true);
  const pieces=path.join('/tmp','fragmentlab-v13-replay.zip'),truth=path.join('/tmp','fragmentlab-v13-truth.zip'),source=path.join('/tmp','fragmentlab-v13-source.png');
  for(const [id,file] of [['downloadFragments',pieces],['downloadTruth',truth],['downloadSource',source]]){
   const [d]=await Promise.all([page.waitForEvent('download'),page.locator('#'+id).click()]);await d.saveAs(file);
  }
  if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-generator.png'),fullPage:true});
  await page.locator('#transfer').click();
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('"pieces":16'),{},{timeout:60000});
  assert.equal(await page.locator('#truthLabel').textContent(),'ground_truth.zip');
  await page.locator('#run').click();await done('Reconstruction exported');
  await page.locator('#resultImg').waitFor({state:'visible'});assert.equal(await page.locator('#placed').textContent(),'16 / 16');assert.equal(await page.locator('#dpa').textContent(),'100.00%');
  if(process.env.EXPECT_CPP)assert.equal(await page.locator('#engineBadge').textContent(),process.env.EXPECT_CPP);
  if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT,fullPage:true});
  await page.locator('#clearTruth').click();await page.locator('#upload').setInputFiles(pieces);
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('"pieces":16'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'—');
  await page.locator('#truthUpload').setInputFiles(truth);await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'100.00%');
  await page.locator('#truthUpload').setInputFiles(source);await page.locator('#run').click();await done('Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'—');
  await page.locator('#comparisonInfo').waitFor({state:'visible'});assert.match(await page.locator('#comparisonInfo').textContent(),/MAE/);
  await page.locator('#imageTab').click();
  assert.equal(await page.locator('#genCount').textContent(),'16 / 16');
  await page.locator('#batch').check();await page.locator('#counts').fill('16, 25');await page.locator('#seeds').fill('42');await page.locator('#profile').selectOption('balanced');await page.locator('#run').click();await done('Benchmark complete');assert.equal(await page.locator('#benchRows tr').count(),2);
  for(const width of [1024,768,390,320]){
   await page.setViewportSize({width,height:844});
   const mobile=await page.evaluate(()=>({viewport:window.innerWidth,scroll:document.documentElement.scrollWidth,overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>window.innerWidth+1}).slice(0,20).map(e=>[e.tagName,e.id,String(e.className),Math.round(e.getBoundingClientRect().right)])}));
   assert.ok(mobile.scroll<=mobile.viewport,JSON.stringify(mobile));
   if(width===390&&process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-mobile.png'),fullPage:true});
  }
  assert.deepEqual(errors,[]);console.log("PASS: generator, transfer, separate truth, blind ZIP replay, image reference, batch, 320–1440 responsive, no JS errors");
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
