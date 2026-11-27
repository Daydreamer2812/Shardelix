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
  // The same reconstruction tab supports opt-in matching without ever
  // treating the generator's evaluation truth as a matching image.
  assert.equal(await page.locator('#useMatchingReference').isChecked(),false);
  assert.equal(await page.locator('#matchReferenceWrap').isHidden(),true);
  await page.locator('#useMatchingReference').check();
  assert.equal(await page.locator('#truthWrap').isHidden(),true);
  assert.equal(await page.locator('#matchReferenceWrap').isVisible(),true);
  assert.equal(await page.locator('#run').isEnabled(),false);
  await page.locator('#profile').selectOption('accurate');
  assert.match(await page.locator('#profileHint').textContent(),/micro-fragment/);
  await page.locator('#matchReferenceUpload').setInputFiles(source);
  assert.equal(await page.locator('#run').isEnabled(),true);
  await page.locator('#useMatchingReference').uncheck();
  await page.locator('#profile').selectOption('balanced');
  assert.equal(await page.locator('#truthWrap').isVisible(),true);
  assert.equal(await page.locator('#matchReferenceWrap').isHidden(),true);
  await page.locator('#run').click();await done('Reconstruction exported');
  await page.locator('#resultImg').waitFor({state:'visible'});assert.equal(await page.locator('#placed').textContent(),'16 / 16');assert.equal(await page.locator('#dpa').textContent(),'100.00%');
  assert.notEqual(await page.locator('#rawDpa').textContent(),'—');
  assert.notEqual(JSON.parse(await page.locator('#diagnostics').textContent()).diagnostics.matching_reference_used,true);
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
  // Folder-wrapped physical strip archive with same-stem foreground masks.
  // Auxiliary masks and order.txt must never count as extra fragments.
  const stripImages=await page.evaluate(()=>Array.from({length:4},(_,i)=>{
    const c=document.createElement('canvas');c.width=19+i;c.height=120;const ctx=c.getContext('2d');
    ctx.fillStyle='black';ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle='white';ctx.fillRect(2,0,14,120);
    const mask=c.toDataURL('image/png').split(',')[1];ctx.fillStyle='#111';ctx.fillRect(5,18+i*8,7,4);
    return [c.toDataURL('image/png').split(',')[1],mask];}));
  const wrapped=path.join('/tmp','shardelix-wrapped-scans.zip');
  execFileSync(process.env.PYTHON||'python',['-c','import sys,zipfile,json,base64; z=zipfile.ZipFile(sys.argv[1],"w"); pairs=json.load(sys.stdin); [(z.writestr(f"collection/{i}.png",base64.b64decode(a)), z.writestr(f"collection/bg_mask_vis/{i}.png",base64.b64decode(b))) for i,(a,b) in enumerate(pairs)]; z.writestr("collection/order.txt","not used"); z.close()',wrapped],{input:JSON.stringify(stripImages)});
  await page.locator('#upload').setInputFiles(wrapped);
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('4 foreground masks recognized'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');
  assert.equal(await page.locator('#placed').textContent(),'4 / 4');
  assert.match(await page.locator('#diagnostics').textContent(),/physical-document-foreground-multiscale/);
  assert.equal(await page.locator('#dpa').textContent(),'—');
  // Metadata-free irregular RGB keyed images: automatic geometry works even
  // after a digital transfer left Input type set to Digital fragments.
  const keyed=await page.evaluate(()=>Array.from({length:102},(_,i)=>{
    const c=document.createElement('canvas');c.width=22+(i%3);c.height=24;
    const ctx=c.getContext('2d');ctx.fillStyle='#ff00ff';ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle=`rgb(${20+(i*37)%200},${20+(i*73)%200},${20+(i*13)%200})`;
    ctx.beginPath();ctx.ellipse(c.width/2,c.height/2,7,9,i%3*.3,0,2*Math.PI);ctx.fill();
    return c.toDataURL('image/png').split(',')[1];
  }));
  const keyedArchive=path.join('/tmp','shardelix-keyed-102.zip');
  execFileSync(process.env.PYTHON||'python',['-c','import sys,zipfile,json,base64; z=zipfile.ZipFile(sys.argv[1],"w"); [(z.writestr(f"collection/fragment_{i}.png",base64.b64decode(v))) for i,v in enumerate(json.load(sys.stdin))]; z.writestr("collection/source_metadata.json",json.dumps(dict(ground_truth="not used"))); z.close()',keyedArchive],{input:JSON.stringify(keyed)});
  await page.locator('#inputMode').selectOption('digital');
  await page.locator('#upload').setInputFiles(keyedArchive);
  await page.waitForFunction(()=>document.querySelector('#scanHelp').hidden===false&&document.querySelector('#metadataInfo').textContent.includes('102'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');
  const largeScan=JSON.parse(await page.locator('#diagnostics').textContent()).diagnostics.scan;
  assert.equal(largeScan.detected_pieces,102);
  assert.equal(largeScan.pair_search,'indexed-contour-color');
  assert.ok(largeScan.searched_pairs<=1020);
  assert.equal(await page.locator('#dpa').textContent(),'—');
  // Exercise the newly exported compact arc kernel in a real WASM worker.
  // Repeated ellipses are deliberately ambiguous: test successful bounded
  // execution and backend selection, not a fictitious correct placement.
  const latticeArchive=path.join('/tmp','shardelix-keyed-lattice-150.zip');
  execFileSync(process.env.PYTHON||'python',['-c','import sys,zipfile,json,base64; z=zipfile.ZipFile(sys.argv[1],"w"); [(z.writestr(f"pieces/part_{i}.png",base64.b64decode(v))) for i,v in enumerate(json.load(sys.stdin))]; z.close()',latticeArchive],{input:JSON.stringify(keyed.concat(keyed.slice(0,48)))});
  await page.locator('#profile').selectOption('fast');
  await page.locator('#upload').setInputFiles(latticeArchive);
  await page.waitForFunction(()=>document.querySelector('#scanHelp').hidden===false&&document.querySelector('#metadataInfo').textContent.includes('150'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');
  const latticeDiag=JSON.parse(await page.locator('#diagnostics').textContent()).diagnostics.scan;
  assert.equal(latticeDiag.detected_pieces,150);
  assert.equal(latticeDiag.lattice.attempted,true);
  if(process.env.EXPECT_CPP)assert.equal(latticeDiag.lattice.arc_backend,'compact-native');
  await page.locator('#profile').selectOption('balanced');
  // Opaque unequal-width image strips: no matte segmentation or invented grid.
  const colorStrips=await page.evaluate(()=>{
    const c=document.createElement('canvas');c.width=60;c.height=150;
    const ctx=c.getContext('2d'),img=ctx.createImageData(c.width,c.height);
    for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){
      const q=(y*c.width+x)*4;
      img.data[q]=120+85*Math.sin(x*.11+y*.037);
      img.data[q+1]=130+80*Math.cos(x*.16-y*.049);
      img.data[q+2]=110+85*Math.sin(x*.071-y*.13);img.data[q+3]=255;
    }ctx.putImageData(img,0,0);
    let x=0;return [11,13,17,19].map((w,i)=>{
      const t=document.createElement('canvas');t.width=i%2?150:w;t.height=i%2?w:150;
      const cx=t.getContext('2d');if(i%2){cx.translate(150,0);cx.rotate(Math.PI/2)}
      cx.drawImage(c,x,0,w,150,0,0,w,150);x+=w;
      return {name:'color-'+i+'.png',data:t.toDataURL('image/png').split(',')[1]};
    });
  });
  await page.locator('#upload').setInputFiles(colorStrips.map(v=>({name:v.name,mimeType:'image/png',buffer:Buffer.from(v.data,'base64')})));
  await page.waitForFunction(()=>document.querySelector('#metadataInfo').textContent.includes('4 image/document strips'),{},{timeout:30000});
  await page.locator('#run').click();await done('Reconstruction exported');
  const stripDiag=JSON.parse(await page.locator('#diagnostics').textContent()).diagnostics;
  assert.equal(stripDiag.input_detection.route,'opaque-strips');
  assert.equal(stripDiag.algorithm,'unknown-width-color-strip-chain');
  assert.equal(stripDiag.placed,4);
  // Text scan fixture: exercise pixel-only classical strip matching.
  const textStrips=await page.evaluate(()=>{
    const c=document.createElement('canvas');c.width=350;c.height=480;
    const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);
    ctx.font='22px Arial';ctx.fillStyle='black';
    for(let y=27;y<470;y+=30)ctx.fillText(y%60===27?'document reconstruction test':'matching paper fragments',4,y);
    return [2,4,1,3,0].map(i=>{
      const tile=document.createElement('canvas');tile.width=70;tile.height=480;
      tile.getContext('2d').drawImage(c,i*70,0,70,480,0,0,70,480);
      return tile.toDataURL('image/png').split(',')[1];
    });
  });
  await page.locator('#inputMode').selectOption('scan');
  assert.equal(await page.locator('#scanOCR').count(),0);
  await page.locator('#upload').setInputFiles(textStrips.map((v,i)=>({name:'scan-'+i+'.png',mimeType:'image/png',buffer:Buffer.from(v,'base64')})));
  await page.locator('#run').click();await done('Reconstruction exported');
  const textDiag=JSON.parse(await page.locator('#diagnostics').textContent()).diagnostics;
  assert.equal(textDiag.algorithm,'physical-document-foreground-multiscale');
  assert.equal(textDiag.scan.kind,'document-strips');
  assert.equal(textDiag.scan.ocr,undefined);
  assert.equal(textDiag.accuracy_verified,false);
  assert.equal(await page.locator('#placed').textContent(),'5 / 5');
  assert.equal(await page.locator('#dpa').textContent(),'—');
  for(const width of [1024,768,390,320]){
   await page.setViewportSize({width,height:844});
   const mobile=await page.evaluate(()=>({viewport:window.innerWidth,scroll:document.documentElement.scrollWidth,overflow:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>window.innerWidth+1}).slice(0,20).map(e=>[e.tagName,e.id,String(e.className),Math.round(e.getBoundingClientRect().right)])}));
   assert.ok(mobile.scroll<=mobile.viewport,JSON.stringify(mobile));
   if(width===390&&process.env.QA_SCREENSHOT)await page.screenshot({path:process.env.QA_SCREENSHOT.replace(/\.png$/,'-mobile.png'),fullPage:true});
  }
  assert.deepEqual(errors,[]);console.log("PASS: generator, transfer, separate truth, blind ZIP replay, image reference, batch, 320–1440 responsive, no JS errors");
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
