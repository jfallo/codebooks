const puppeteer = require('puppeteer');
const Papa = require("papaparse");
const fs = require('fs');
const path = require('path');
const { error } = require('console');
const downloadPath = 'intermediate/';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function getStudiesPages(page) {
    const studiesPages = [];

    // navigate through ICPSR studies database
    for (let start = 0; start <= 6591; start += 50) {
        const url = `https://www.icpsr.umich.edu/web/ICPSR/search/studies?start=${start}&sort=score%20desc%2CTITLE_SORT%20asc&OWNER=ICPSR&ARCHIVE=ICPSR&ARCHIVE=ICPSR&ARCHIVE=ICPSR&PUBLISH_STATUS=PUBLISHED&rows=50&q=`
        await page.goto(url);
        await page.waitForSelector('.searchResult a[href*="/studies/"]');

        // get links to studies
        links = await page.$$eval(
            '.searchResult a[href*="/studies/"]',
            as => as.map(a => a.href)
        );

        // push them to list of studies pages
        studiesPages.push(...links);
    }

    // save studies to indexed csv
    const csvLines = [
        'index,url',
        ...studiesPages.map((url, i) => `${i+1},${url}`)
    ];
    fs.writeFileSync('intermediate/studies_pages.csv', csvLines.join('\n'));
    await sleep(1000);
}

async function getCodebook(study, studyPage, csvLines, page) {
    try {
        await page.goto(studyPage + '/datadocumentation');
        await sleep(2000);

        const studyName = await page.$eval('h1.study-title', el => el.textContent.trim());

        let i = 0;
        while (true) {
            const rowCount = await page.$$eval('#datadocumentationTab tr', rows => rows.length);
            if (i >= rowCount) break;

            const rows = await page.$$('#datadocumentationTab tr');
            const row = rows[i];
            i++;
            if (!row) continue;

            const data = await row.$('td');
            if (!data) continue;

            const docName = await page.evaluate(data => {
                const spans = data.querySelectorAll('span');
                return Array.from(spans).map(s => s.textContent.trim()).join(' ').replace(/\s+/g, ' ');
            }, data);

            const buttonCount = await row.$$eval('button.dropdown-toggle', btns => btns.length).catch(() => 0);

            for (let j = 0; j < buttonCount; j++) {
                const freshRows = await page.$$('#datadocumentationTab tr');
                const freshRow = freshRows[i - 1]; // i already incremented
                if (!freshRow) continue;

                const buttons = await freshRow.$$('button.dropdown-toggle');
                const button = buttons[j];
                if (!button) continue;

                const hasDownloadIcon = await page.evaluate(btn =>
                    !!btn.querySelector('.fa-download'), button
                );
                if (!hasDownloadIcon) continue;

                await page.keyboard.press('Escape');
                await sleep(300);

                await button.click();
                await page.waitForSelector('ul.dropdown-menu.download-link-menu.show', { visible: true });
                await sleep(300);

                const linkTexts = await page.$$eval(
                    'ul.dropdown-menu.download-link-menu.show a.doc.dropdown-item',
                    links => links.map(l => ({ text: l.textContent.trim(), href: l.href }))
                );

                console.log('Links found:', linkTexts.map(l => l.text));

                const codebook = linkTexts.find(l => l.text.includes('Codebook [PDF]'));
                if (codebook) {
                    csvLines.push(`${study}|${studyPage}|${studyName}|${docName}|${codebook.text}`);
                    console.log(`Downloading: ${study}|${docName}`);

                    const links = await page.$$('ul.dropdown-menu.download-link-menu.show a.doc.dropdown-item');
                    let codebookLink = null;
                    for (const l of links) {
                        const text = await page.evaluate(el => el.textContent.trim(), l);
                        if (text.includes('Codebook [PDF]')) {
                            codebookLink = l;
                            break;
                        }
                    }

                    if (codebookLink) {
                        await codebookLink.click();
                        await sleep(2000);
                    }

                    let navBack = false;
                    while (!navBack) {
                        try {
                            await page.goto(studyPage + '/datadocumentation');
                            await sleep(2000);
                            navBack = true;
                        } catch (error) {
                            await sleep(2000);
                        }
                    }
                    break; 
                } else {
                    await page.keyboard.press('Escape');
                    await sleep(300);
                }
            }
        }

        return true;
    } catch (err) {
        console.error(err.message);
        console.log('Trying again...');
        return false;
    }
}

async function main() {
    const csvLines = ['index|url|study|data|codebook'];
    
    // open browser and set downloads path
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    const client = await page.createCDPSession();
    await client.send('Page.setDownloadBehavior', {
        behavior: 'allow',
        downloadPath: './intermediate'
    });
    
    // get studies
    await getStudiesPages(page);
    const studiesPages = fs.readFileSync('intermediate/studies_pages.csv', 'utf-8')
        .split('\n');
    studiesPages.shift();

    // for each study...
    for (const studyPage of studiesPages) {
        const i = studyPage.split(',')[0];
        const url = studyPage.split(',')[1];
        let res = false;

        // download codebook if it is not there
        const csv = fs.readFileSync('./output/codebooks_metadata.csv', 'utf8');
        const parsed = Papa.parse(csv, {
            header: true,
            delimiter: '|'
        });

        const df = parsed.data;
        if (df.some(row => row.url === url)) {
            res = df.some(row => row.url === url && row.codebook && row.codebook.trim() !== '');

            while (!res) 
                res = await getCodebook(i, url, csvLines, page);

            fs.writeFileSync('intermediate/codebooks_metadata.csv', csvLines.join('\n'), 'utf8');
        }
    }

    // close browser
    await browser.close();
}

main().catch(err => console.error(err));