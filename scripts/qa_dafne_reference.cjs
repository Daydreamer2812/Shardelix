/* Native Studio browser integration, using official separated DAFNE inputs.
 * node scripts/qa_dafne_reference.cjs URL input.zip original.jpg output.zip
 */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const [url,input,reference,output]=process.argv.slice(2);
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({acceptDownloads:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#fragmentTab').click();
  await page.locator('#useMatchingReference').check();
  await page.locator('#upload').setInputFiles(input);
  assert.equal(await page.locator('#run').isEnabled(),false);
  assert.match(await page.locator('#preflight').textContent(),/matching reference/);
  assert.equal(await page.locator('#truthWrap').isHidden(),true);
  await page.locator('#matchReferenceUpload').setInputFiles(reference);
  assert.equal(await page.locator('#geometryFields').isHidden(),true);
  assert.equal(await page.locator('#run').isEnabled(),true);
  await page.locator('#run').click();
  await page.waitForFunction(()=>document.querySelector('#stage').textContent==='Reconstruction exported'||document.querySelector('#dot').classList.contains('error'),{},{timeout:300000});
  assert.equal(await page.locator('#stage').textContent(),'Reconstruction exported',await page.locator('#error').textContent());
  await page.locator('#resultImg').waitFor({state:'visible'});
  assert.equal(await page.locator('#placed').textContent(),'87 / 102');
  assert.equal(await page.locator('#dpa').textContent(),'—');
  assert.match(await page.locator('#orientationLabel').textContent(),/Reference-assisted.*not blind/);
  assert.match(await page.locator('#notice').textContent(),/No ground truth used/);
  const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Results ZIP',exact:true}).click()]);
  await download.saveAs(output);
  assert.deepEqual(errors,[]);
  console.log('PASS native Studio Chromium upload, reference mode, 87/102 accepted, export, no misleading blind/DPA labels');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
