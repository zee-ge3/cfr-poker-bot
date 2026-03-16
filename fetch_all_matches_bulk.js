// === CMU AI Poker — Bulk Match Log Fetcher ===
// Run in browser console at https://aipoker.cmudsc.com/dashboard
//
// Downloads ALL match CSVs into a single JSON bundle file,
// then use: python3 import_matches.py match_bundle.json
//
// Usage:
//   1. Open https://aipoker.cmudsc.com/dashboard
//   2. F12 → Console → paste this → Enter
//   3. Wait ~2-5 minutes for all matches to be fetched
//   4. A single file 'match_bundle_TIMESTAMP.json' will download
//   5. Move it to the poker-engine-2026 directory
//   6. Run: python3 import_matches.py match_bundle_TIMESTAMP.json

(async () => {
  const ROWS = 100;
  const TEAM_ID = 71;
  const DELAY = 250;
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function parseTurboStream(raw) {
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return null;
    if (arr[0] === 'SingleFetchRedirect' || (Array.isArray(arr[0]) && arr[0][0] === 'SingleFetchRedirect')) return null;
    const cache = {};
    function resolve(idx, d = 0) {
      if (d > 50) return null;
      if (typeof idx !== 'number') return idx;
      if (idx === -5) return null;
      if (idx < 0) return idx;
      if (idx in cache) return cache[idx];
      if (idx >= arr.length) return null;
      const v = arr[idx];
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        const r = {}; cache[idx] = r;
        for (const [k, val] of Object.entries(v)) {
          const key = resolve(parseInt(k.replace(/^_/, '')), d+1);
          if (typeof key === 'string') r[key] = resolve(val, d+1);
        }
        return r;
      } else if (Array.isArray(v)) {
        const r = []; cache[idx] = r;
        for (const i of v) r.push(resolve(i, d+1));
        return r;
      }
      cache[idx] = v; return v;
    }
    const root = resolve(0);
    if (!root || typeof root !== 'object') return null;
    for (const k of ['routes/dashboard','dashboard']) {
      if (root[k]?.data) return root[k].data;
      if (root[k]) return root[k];
    }
    return root;
  }

  async function getLogUrl(matchId, type, teamId) {
    const url = new URL(`/api/logs/${matchId}/${type}`, location.origin);
    if (type === 'bot' && teamId) url.searchParams.set('teamId', String(teamId));
    const r = await fetch(url);
    if (!r.ok) return null;
    return (await r.json().catch(() => null))?.url || null;
  }

  async function downloadText(url) {
    try { const r = await fetch(url); return r.ok ? await r.text() : null; }
    catch { return null; }
  }

  function saveFile(name, content) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], {type:'application/json'}));
    a.download = name; document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(a.href);
  }

  console.log('🃏 Fetching all match history...');
  let allMatches = [], page = 1;
  while (true) {
    const url = `/dashboard.data?page=${page}&rows=${ROWS}&viewportHeight=900&viewportWidth=1400`;
    const resp = await fetch(url);
    const data = parseTurboStream(await resp.text());
    if (!data) { if (page === 1) { console.error('❌ Not logged in'); return; } break; }
    const matches = data.matches || [];
    if (page === 1) console.log(`  ${data.totalMatches} matches, ${data.totalPages} pages`);
    allMatches.push(...matches);
    if (page >= (data.totalPages || 1)) break;
    page++; await sleep(DELAY);
  }

  console.log(`\n📥 Downloading ${allMatches.length} match logs...`);
  const bundle = { team: 'geoz', teamId: TEAM_ID, fetchedAt: new Date().toISOString(), matches: [] };
  let done = 0, errors = 0;

  for (const m of allMatches) {
    const matchId = m.matchId || m.match?.id;
    const teamId = m.teamId || TEAM_ID;
    if (!matchId) continue;
    const status = m.match?.matchStatus || '';
    if (status !== 'COMPLETED' && status !== 'ERROR' && status !== '') continue;

    const opp = m.match?.teamMatches?.find(t => t.teamId !== teamId)?.team?.name || '?';
    const entry = {
      matchId, teamId, bankroll: m.bankroll, eloDelta: m.eloDelta,
      timeUsed: m.timeUsed, botName: m.bot?.name, opponent: opp,
      status, timestamp: m.match?.timestamp, matchType: m.match?.matchType,
      csv: null, botLog: null,
    };

    try {
      // Get CSV
      const csvUrl = await getLogUrl(matchId, 'csv');
      if (csvUrl) entry.csv = await downloadText(csvUrl);
      await sleep(150);

      // Get bot log
      const botUrl = await getLogUrl(matchId, 'bot', teamId);
      if (botUrl) entry.botLog = await downloadText(botUrl);
      await sleep(150);

      done++;
      if (done % 10 === 0) console.log(`  ${done}/${allMatches.length}...`);
    } catch (e) {
      console.warn(`  ❌ ${matchId}: ${e.message}`);
      errors++;
    }
    bundle.matches.push(entry);
  }

  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `match_bundle_${ts}.json`;
  saveFile(filename, JSON.stringify(bundle));
  console.log(`\n✅ Done! ${done} matches fetched (${errors} errors)`);
  console.log(`📦 Saved ${filename}`);
  console.log(`\nNext: mv ~/Downloads/${filename} ~/poker/poker-engine-2026/`);
  console.log(`Then: python3 import_matches.py ${filename}`);
})();
