/* Optional Playwright QA: native UI single run, batch, metadata replay, responsive layout.
   Run with a local Studio already listening: node scripts/qa_studio.cjs http://127.0.0.1:8765
   npm install playwright is only needed in environments that do not supply it.
*/
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8765', browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true}), errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base); await page.locator('#sample').click();await page.locator('#count').fill('16');await page.locator('#profile').selectOption('fast');
 await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#stage').textContent==='Reconstruction exported'||document.querySelector('#dot').classList.contains('error'),{},{timeout:300000});
 assert.equal(await page.locator('#stage').textContent(),'Reconstruction exported');
 await page.locator('#resultImg').waitFor({state:'visible'});assert.equal(await page.locator('#placed').textContent(),'16 / 16');
 const file=path.join('/tmp','fragmentlab-v12-ui-replay.zip');
 const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Shuffled pieces ZIP'}).click()]); await download.saveAs(file);
 if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT,fullPage:true});
 await page.locator('#batch').check();await page.locator('#counts').fill('16, 25');await page.locator('#seeds').fill('42');await page.locator('#profile').selectOption('balanced');await page.locator('#run').click();
 await page.waitForFunction(()=>document.querySelector('#stage').textContent==='Benchmark complete'||document.querySelector('#dot').classList.contains('error'),{},{timeout:300000});
 assert.equal(await page.locator('#stage').textContent(),'Benchmark complete');
 assert.equal(await page.locator('#benchRows tr').count(),2);
 await page.locator('#fragmentTab').click();await page.locator('#upload').setInputFiles(file);
 await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('"pieces":16'),{},{timeout:30000});
 assert.equal(await page.locator('#geometryFields').isHidden(),true);
 await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#stage').textContent==='Reconstruction exported'||document.querySelector('#dot').classList.contains('error'),{},{timeout:300000});
 assert.equal(await page.locator('#stage').textContent(),'Reconstruction exported');assert.equal(await page.locator('#dpa').textContent(),'—');
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>({viewport:window.innerWidth,scroll:document.documentElement.scrollWidth,
   overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>window.innerWidth+1}).slice(0,20).map(e=>({element:e.tagName,id:e.id,cls:String(e.className),right:Math.round(e.getBoundingClientRect().right)}))}));
 if(process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-mobile.png'),fullPage:true});
 assert.ok(mobile.scroll<=mobile.viewport,JSON.stringify(mobile));
 assert.deepEqual(errors,[]);console.log("PASS: single, batch, metadata replay, responsive layout, no JS errors");
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
