// === CMU AI Poker — Fetch All Match Logs ===
// Run this in your browser console at https://aipoker.cmudsc.com/dashboard
//
// It will:
//   1. Fetch all match history pages from the dashboard API
//   2. For each match, get the presigned S3 URLs for CSV and bot logs
//   3. Download all CSVs and bot logs as files
//
// Usage:
//   1. Open https://aipoker.cmudsc.com/dashboard
//   2. Open DevTools (F12) → Console
//   3. Paste this entire script and press Enter
//   4. Wait for all downloads to complete
//   5. Move downloaded files to tournament_logs/

(async () => {
  const ROWS = 100;
  const TEAM_ID = 71; // geoz
  const DELAY_MS = 300; // delay between API calls to avoid rate limiting

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // Fetch a page of matches from the dashboard
  async function fetchPage(page) {
    const url = `/dashboard.data?page=${page}&rows=${ROWS}&viewportHeight=900&viewportWidth=1400`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Dashboard fetch failed: ${resp.status}`);
    const raw = await resp.text();
    return parseTurboStream(raw);
  }

  // Parse Remix v3 turbo-stream format
  function parseTurboStream(raw) {
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return null;

    // Check for redirect
    if (arr.length > 0 && arr[0] === 'SingleFetchRedirect') return null;
    if (arr.length > 0 && Array.isArray(arr[0]) && arr[0][0] === 'SingleFetchRedirect') return null;

    const cache = {};
    function resolve(idx, depth = 0) {
      if (depth > 50) return null;
      if (typeof idx === 'number') {
        if (idx === -5) return null;
        if (idx < 0) return idx;
        if (idx in cache) return cache[idx];
        if (idx >= arr.length) return null;
        const val = arr[idx];
        if (val && typeof val === 'object' && !Array.isArray(val)) {
          const result = {};
          cache[idx] = result;
          for (const [k, v] of Object.entries(val)) {
            const keyIdx = parseInt(k.replace(/^_/, ''));
            const key = resolve(keyIdx, depth + 1);
            const value = resolve(v, depth + 1);
            if (typeof key === 'string') result[key] = value;
          }
          return result;
        } else if (Array.isArray(val)) {
          const result = [];
          cache[idx] = result;
          for (const item of val) result.push(resolve(item, depth + 1));
          return result;
        } else {
          cache[idx] = val;
          return val;
        }
      }
      return idx;
    }

    const root = resolve(0);
    if (!root || typeof root !== 'object') return null;

    // Find the dashboard route data
    for (const key of ['routes/dashboard', 'dashboard']) {
      if (root[key]?.data) return root[key].data;
      if (root[key]) return root[key];
    }
    return root;
  }

  // Get presigned URL for a match log
  async function getLogUrl(matchId, type, teamId) {
    const url = new URL(`/api/logs/${matchId}/${type}`, window.location.origin);
    if (type === 'bot' && teamId) url.searchParams.set('teamId', teamId.toString());
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const data = await resp.json().catch(() => null);
    return data?.url || null;
  }

  // Download content from URL and return as text
  async function downloadContent(url) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      return await resp.text();
    } catch (e) {
      console.warn('Download failed:', e);
      return null;
    }
  }

  // Save text content as a downloadable file
  function saveFile(filename, content) {
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  // === Alternative: collect all data as JSON for bulk processing ===
  async function collectMatchData() {
    console.log('📊 Fetching match history...');
    let allMatches = [];
    let page = 1;

    while (true) {
      console.log(`  Page ${page}...`);
      const data = fetchPage ? await fetchPage(page) : null;
      if (!data) {
        if (page === 1) {
          console.error('❌ Auth failed or not on dashboard. Make sure you are logged in.');
          return null;
        }
        break;
      }

      const matches = data.matches || [];
      const totalPages = data.totalPages || 1;
      const totalMatches = data.totalMatches || 0;

      if (page === 1) {
        console.log(`  Found ${totalMatches} total matches across ${totalPages} pages`);
      }

      allMatches = allMatches.concat(matches);

      if (page >= totalPages) break;
      page++;
      await sleep(DELAY_MS);
    }

    return allMatches;
  }

  // Main execution
  console.log('🃏 CMU AI Poker — Match Log Fetcher');
  console.log('====================================\n');

  const matches = await collectMatchData();
  if (!matches) return;

  console.log(`\n📥 Downloading logs for ${matches.length} matches...`);
  console.log('   (This may take a few minutes)\n');

  let downloaded = 0;
  let failed = 0;

  for (const m of matches) {
    const matchId = m.matchId || m.match?.id;
    const teamId = m.teamId || TEAM_ID;
    if (!matchId) continue;

    const status = m.match?.matchStatus || '';
    if (status !== 'COMPLETED' && status !== 'ERROR' && status !== '') continue;

    // Find opponent name
    const opp = m.match?.teamMatches?.find(tm => tm.teamId !== teamId)?.team?.name || 'unknown';
    const bankroll = m.bankroll || 0;
    const result = bankroll > 0 ? 'W' : 'L';

    // Download CSV
    try {
      const csvUrl = await getLogUrl(matchId, 'csv');
      if (csvUrl) {
        const content = await downloadContent(csvUrl);
        if (content) {
          saveFile(`match_${matchId}.csv`, content);
          downloaded++;
        }
      }
      await sleep(200);

      // Download bot log
      const botUrl = await getLogUrl(matchId, 'bot', teamId);
      if (botUrl) {
        const content = await downloadContent(botUrl);
        if (content) {
          const botName = m.bot?.name || 'bot';
          saveFile(`match_${matchId}_bot_${botName}.log`, content);
        }
      }
      await sleep(200);

      console.log(`  ✅ ${matchId} vs ${opp} [${result}${bankroll >= 0 ? '+' : ''}${bankroll}]`);
    } catch (e) {
      console.warn(`  ❌ ${matchId}: ${e.message}`);
      failed++;
    }
  }

  console.log(`\n✅ Done! Downloaded ${downloaded} CSVs (${failed} failed)`);
  console.log('Move files from Downloads to tournament_logs/');

  // Also save a JSON manifest of all matches for offline analysis
  const manifest = matches.map(m => ({
    matchId: m.matchId,
    teamId: m.teamId,
    bankroll: m.bankroll,
    eloDelta: m.eloDelta,
    timeUsed: m.timeUsed,
    botName: m.bot?.name,
    opponent: m.match?.teamMatches?.find(tm => tm.teamId !== (m.teamId || TEAM_ID))?.team?.name,
    status: m.match?.matchStatus,
    timestamp: m.match?.timestamp,
    matchType: m.match?.matchType,
  }));
  saveFile('match_manifest.json', JSON.stringify(manifest, null, 2));
  console.log('📋 Saved match_manifest.json with all match metadata');
})();
