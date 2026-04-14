/**
 * DimensionalBase Dashboard
 * Pure vanilla JS — no build step, no dependencies.
 */

class DashboardApp {
  constructor() {
    // State
    this.entries = [];
    this.events = [];
    this.trustData = {};
    this.ws = null;
    this.wsReconnectDelay = 1000;
    this.wsMaxReconnectDelay = 30000;
    this.wsReconnectTimer = null;
    this.statusTimer = null;
    this.expandedRowPath = null;

    // Cache DOM references
    this.dom = {
      // Header
      wsIndicator: document.getElementById("wsIndicator"),
      wsLabel: document.getElementById("wsLabel"),
      versionBadge: document.getElementById("versionBadge"),

      // Sidebar
      sidebar: document.getElementById("sidebar"),
      sidebarToggle: document.getElementById("sidebarToggle"),
      statEntries: document.getElementById("statEntries"),
      statOwners: document.getElementById("statOwners"),
      statNamespaces: document.getElementById("statNamespaces"),
      statConflicts: document.getElementById("statConflicts"),
      statGaps: document.getElementById("statGaps"),
      statStale: document.getElementById("statStale"),
      sidebarTrust: document.getElementById("sidebarTrust"),

      // Knowledge
      knowledgeSearch: document.getElementById("knowledgeSearch"),
      knowledgeBody: document.getElementById("knowledgeBody"),
      entryCount: document.getElementById("entryCount"),

      // Events
      eventFeed: document.getElementById("eventFeed"),
      eventCount: document.getElementById("eventCount"),
      eventPlaceholder: document.getElementById("eventPlaceholder"),
      clearEvents: document.getElementById("clearEvents"),

      // Trust
      trustChart: document.getElementById("trustChart"),

      // Provenance
      provenancePath: document.getElementById("provenancePath"),
      provenanceLookup: document.getElementById("provenanceLookup"),
      provenanceResult: document.getElementById("provenanceResult"),
    };

    this._bindEvents();
    this._initTabs();
    this._connectWebSocket();
    this._refreshStatus();
    this._refreshEntries();
    this._refreshTrust();
    this._refreshEventHistory();

    // Auto-refresh status every 5 seconds
    this.statusTimer = setInterval(() => this._refreshStatus(), 5000);
  }

  // ──────────────────────────────────────────────
  //  API helpers
  // ──────────────────────────────────────────────

  /**
   * Fetch wrapper with error handling.
   * Returns parsed JSON or null on failure.
   */
  async _api(path, options = {}) {
    try {
      const resp = await fetch(path, {
        ...options,
        headers: { "Accept": "application/json", ...(options.headers || {}) },
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        console.warn(`API ${resp.status} ${path}: ${body}`);
        return null;
      }
      return await resp.json();
    } catch (err) {
      console.warn(`API error ${path}:`, err);
      return null;
    }
  }

  // ──────────────────────────────────────────────
  //  WebSocket
  // ──────────────────────────────────────────────

  _connectWebSocket() {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}/ws/subscribe`;

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.warn("WebSocket construction failed:", err);
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.wsReconnectDelay = 1000;
      this._setConnected(true);
    };

    this.ws.onclose = () => {
      this._setConnected(false);
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      this._setConnected(false);
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        // Skip ack messages
        if (data.ack !== undefined) return;
        this._handleEvent(data);
      } catch (err) {
        console.warn("WS message parse error:", err);
      }
    };
  }

  _scheduleReconnect() {
    if (this.wsReconnectTimer) return;
    this.wsReconnectTimer = setTimeout(() => {
      this.wsReconnectTimer = null;
      this.wsReconnectDelay = Math.min(this.wsReconnectDelay * 1.5, this.wsMaxReconnectDelay);
      this._connectWebSocket();
    }, this.wsReconnectDelay);
  }

  _setConnected(connected) {
    if (connected) {
      this.dom.wsIndicator.classList.add("connected");
      this.dom.wsLabel.textContent = "Connected";
      this.dom.wsIndicator.title = "WebSocket connected";
    } else {
      this.dom.wsIndicator.classList.remove("connected");
      this.dom.wsLabel.textContent = "Disconnected";
      this.dom.wsIndicator.title = "WebSocket disconnected";
    }
  }

  // ──────────────────────────────────────────────
  //  Event handling
  // ──────────────────────────────────────────────

  _handleEvent(data) {
    this.events.unshift(data);
    // Cap at 200 events in memory
    if (this.events.length > 200) {
      this.events.length = 200;
    }
    this._prependEventItem(data);
    this._updateEventCount();

    // If it's a CHANGE event, refresh entries and status
    if (data.type === "CHANGE") {
      this._refreshEntries();
      this._refreshStatus();
      this._refreshTrust();
    }
  }

  _prependEventItem(data) {
    // Remove placeholder
    if (this.dom.eventPlaceholder) {
      this.dom.eventPlaceholder.remove();
      this.dom.eventPlaceholder = null;
    }

    const item = document.createElement("div");
    item.className = "event-item";
    item.setAttribute("data-type", data.type || "CHANGE");

    const ts = data.timestamp ? this._formatTime(data.timestamp) : "";
    const eventData = typeof data.data === "object" ? JSON.stringify(data.data) : (data.data || "");

    item.innerHTML = `
      <span class="event-type">${this._esc(data.type || "EVENT")}</span>
      <div class="event-body">
        <div class="event-path">${this._esc(data.path || "")}</div>
        ${eventData ? `<div class="event-data" title="${this._esc(eventData)}">${this._esc(eventData)}</div>` : ""}
      </div>
      <div class="event-meta">
        <span>${this._esc(ts)}</span>
        ${data.source_owner ? `<span>${this._esc(data.source_owner)}</span>` : ""}
      </div>
    `;

    this.dom.eventFeed.prepend(item);

    // Cap rendered items at 150
    while (this.dom.eventFeed.children.length > 150) {
      this.dom.eventFeed.lastElementChild.remove();
    }
  }

  _updateEventCount() {
    this.dom.eventCount.textContent = `${this.events.length} event${this.events.length !== 1 ? "s" : ""}`;
  }

  // ──────────────────────────────────────────────
  //  Status
  // ──────────────────────────────────────────────

  async _refreshStatus() {
    const data = await this._api("/api/v1/status");
    if (!data) return;

    this.dom.statEntries.textContent = data.total_entries ?? data.entries ?? "\u2014";
    this.dom.statOwners.textContent = data.unique_owners ?? data.owners ?? "\u2014";
    this.dom.statNamespaces.textContent = data.namespaces ?? data.unique_namespaces ?? "\u2014";
    this.dom.statConflicts.textContent = data.active_conflicts ?? data.conflicts ?? 0;
    this.dom.statGaps.textContent = data.active_gaps ?? data.gaps ?? 0;
    this.dom.statStale.textContent = data.stale_entries ?? data.stale ?? 0;
  }

  // ──────────────────────────────────────────────
  //  Knowledge Entries
  // ──────────────────────────────────────────────

  async _refreshEntries() {
    const data = await this._api("/api/v1/entries?scope=**&budget=50000");
    if (!data || !data.entries) return;

    this.entries = data.entries;
    this._renderEntries();
  }

  _renderEntries(filter = "") {
    const filt = filter.toLowerCase().trim();
    const filtered = filt
      ? this.entries.filter(e =>
          (e.path || "").toLowerCase().includes(filt) ||
          (e.value || "").toLowerCase().includes(filt) ||
          (e.owner || "").toLowerCase().includes(filt)
        )
      : this.entries;

    this.dom.entryCount.textContent = `${filtered.length} of ${this.entries.length} entries`;

    if (filtered.length === 0) {
      this.dom.knowledgeBody.innerHTML = `<tr><td colspan="5" class="muted">${
        this.entries.length === 0 ? "No entries in database." : "No entries match filter."
      }</td></tr>`;
      return;
    }

    const fragment = document.createDocumentFragment();

    for (const entry of filtered) {
      const tr = document.createElement("tr");
      tr.setAttribute("data-path", entry.path);

      const conf = (entry.confidence != null) ? entry.confidence.toFixed(2) : "\u2014";
      const confColor = entry.confidence >= 0.8 ? "var(--green)"
                      : entry.confidence >= 0.5 ? "var(--amber)"
                      : "var(--red)";

      tr.innerHTML = `
        <td><span class="cell-path">${this._esc(entry.path)}</span></td>
        <td><span class="cell-value" title="${this._esc(entry.value)}">${this._esc(this._truncate(entry.value, 80))}</span></td>
        <td><span class="cell-owner">${this._esc(entry.owner)}</span></td>
        <td><span class="cell-type">${this._esc(entry.type)}</span></td>
        <td><span class="cell-confidence" style="color:${confColor}">${conf}</span></td>
      `;

      tr.addEventListener("click", () => this._toggleEntryDetail(entry, tr));
      fragment.appendChild(tr);

      // If this row was expanded, re-expand it
      if (this.expandedRowPath === entry.path) {
        const detailTr = this._createDetailRow(entry);
        fragment.appendChild(detailTr);
      }
    }

    this.dom.knowledgeBody.innerHTML = "";
    this.dom.knowledgeBody.appendChild(fragment);
  }

  _toggleEntryDetail(entry, rowEl) {
    // If already expanded, collapse
    if (this.expandedRowPath === entry.path) {
      this.expandedRowPath = null;
      const next = rowEl.nextElementSibling;
      if (next && next.classList.contains("detail-row")) {
        next.remove();
      }
      return;
    }

    // Collapse any previously expanded row
    const prev = this.dom.knowledgeBody.querySelector(".detail-row");
    if (prev) prev.remove();
    this.expandedRowPath = entry.path;

    const detailTr = this._createDetailRow(entry);
    rowEl.after(detailTr);
  }

  _createDetailRow(entry) {
    const tr = document.createElement("tr");
    tr.className = "detail-row";

    const refs = (entry.refs && entry.refs.length) ? entry.refs.join(", ") : "none";
    const created = entry.created_at ? new Date(entry.created_at * 1000).toLocaleString() : "\u2014";
    const updated = entry.updated_at ? new Date(entry.updated_at * 1000).toLocaleString() : "\u2014";
    const meta = entry.metadata ? JSON.stringify(entry.metadata) : "{}";

    tr.innerHTML = `
      <td colspan="5">
        <div class="detail-content">
          <div class="detail-field"><span class="detail-label">Full Value</span><span class="detail-val">${this._esc(entry.value)}</span></div>
          <div class="detail-field"><span class="detail-label">ID</span><span class="detail-val">${this._esc(entry.id)}</span></div>
          <div class="detail-field"><span class="detail-label">Version</span><span class="detail-val">${entry.version}</span></div>
          <div class="detail-field"><span class="detail-label">TTL</span><span class="detail-val">${this._esc(entry.ttl)}</span></div>
          <div class="detail-field"><span class="detail-label">Refs</span><span class="detail-val">${this._esc(refs)}</span></div>
          <div class="detail-field"><span class="detail-label">Created</span><span class="detail-val">${created}</span></div>
          <div class="detail-field"><span class="detail-label">Updated</span><span class="detail-val">${updated}</span></div>
          <div class="detail-field"><span class="detail-label">Score</span><span class="detail-val">${(entry.score || 0).toFixed(4)}</span></div>
          <div class="detail-field"><span class="detail-label">Metadata</span><span class="detail-val">${this._esc(meta)}</span></div>
        </div>
      </td>
    `;
    return tr;
  }

  // ──────────────────────────────────────────────
  //  Trust
  // ──────────────────────────────────────────────

  async _refreshTrust() {
    const data = await this._api("/api/v1/trust");
    if (!data) return;
    this.trustData = data;
    this._renderTrustChart(data);
    this._renderSidebarTrust(data);
  }

  _renderTrustChart(data) {
    // data can be { agents: { name: {score: ...} } } or { name: score } — handle both
    const agents = this._normalizeTrustData(data);

    if (agents.length === 0) {
      this.dom.trustChart.innerHTML = '<p class="muted">No agent trust data available.</p>';
      return;
    }

    // Sort by score descending
    agents.sort((a, b) => b.score - a.score);

    const fragment = document.createDocumentFragment();

    for (const agent of agents) {
      const pct = Math.max(0, Math.min(100, agent.score * 100));
      const cls = pct >= 70 ? "score-high" : pct >= 40 ? "score-mid" : "score-low";

      const row = document.createElement("div");
      row.className = "trust-row";
      row.innerHTML = `
        <span class="trust-agent-name" title="${this._esc(agent.name)}">${this._esc(agent.name)}</span>
        <div class="trust-bar-track">
          <div class="trust-bar-fill ${cls}" style="width: 0%"></div>
        </div>
        <span class="trust-score-label">${pct.toFixed(1)}%</span>
      `;
      fragment.appendChild(row);
    }

    this.dom.trustChart.innerHTML = "";
    this.dom.trustChart.appendChild(fragment);

    // Animate bars after paint
    requestAnimationFrame(() => {
      this.dom.trustChart.querySelectorAll(".trust-bar-fill").forEach((bar, i) => {
        const pct = Math.max(0, Math.min(100, agents[i].score * 100));
        bar.style.width = pct + "%";
      });
    });
  }

  _renderSidebarTrust(data) {
    const agents = this._normalizeTrustData(data);

    if (agents.length === 0) {
      this.dom.sidebarTrust.innerHTML = '<p class="muted">No data</p>';
      return;
    }

    agents.sort((a, b) => b.score - a.score);
    // Show top 8 in sidebar
    const top = agents.slice(0, 8);

    const fragment = document.createDocumentFragment();
    for (const agent of top) {
      const pct = Math.max(0, Math.min(100, agent.score * 100));
      const div = document.createElement("div");
      div.className = "sidebar-trust-item";
      div.innerHTML = `
        <div class="sidebar-trust-label">
          <span class="sidebar-trust-name" title="${this._esc(agent.name)}">${this._esc(agent.name)}</span>
          <span class="sidebar-trust-score">${pct.toFixed(0)}%</span>
        </div>
        <div class="sidebar-trust-bar">
          <div class="sidebar-trust-fill" style="width: ${pct}%"></div>
        </div>
      `;
      fragment.appendChild(div);
    }

    this.dom.sidebarTrust.innerHTML = "";
    this.dom.sidebarTrust.appendChild(fragment);
  }

  /**
   * Normalize various trust response shapes into [{name, score}].
   */
  _normalizeTrustData(data) {
    if (!data) return [];

    // Shape: { agents: { "name": { score: 0.9, ... }, ... } }
    if (data.agents && typeof data.agents === "object") {
      return Object.entries(data.agents).map(([name, val]) => ({
        name,
        score: typeof val === "number" ? val : (val.score ?? val.trust_score ?? 0),
      }));
    }

    // Shape: [ { agent: "name", score: 0.9 }, ... ]
    if (Array.isArray(data)) {
      return data.map(item => ({
        name: item.agent || item.name || item.owner || "unknown",
        score: item.score ?? item.trust_score ?? 0,
      }));
    }

    // Shape: { "name": 0.9, "name2": 0.8, ... } (flat object)
    if (typeof data === "object") {
      // Filter out non-agent keys
      const skip = new Set(["total_agents", "average_trust", "trust_model"]);
      return Object.entries(data)
        .filter(([k, v]) => !skip.has(k) && (typeof v === "number" || (typeof v === "object" && v !== null)))
        .map(([name, val]) => ({
          name,
          score: typeof val === "number" ? val : (val.score ?? val.trust_score ?? 0),
        }));
    }

    return [];
  }

  // ──────────────────────────────────────────────
  //  Event History
  // ──────────────────────────────────────────────

  async _refreshEventHistory() {
    const data = await this._api("/api/v1/events");
    if (!data) return; // Endpoint may not exist yet — that's fine

    const events = Array.isArray(data) ? data : (data.events || []);
    // Oldest first in the array, we want newest first
    for (const evt of events.reverse()) {
      this.events.push(evt);
      this._prependEventItem(evt);
    }
    this._updateEventCount();
  }

  // ──────────────────────────────────────────────
  //  Provenance
  // ──────────────────────────────────────────────

  async _lookupProvenance() {
    const path = (this.dom.provenancePath.value || "").trim();
    if (!path) {
      this.dom.provenanceResult.innerHTML = '<p class="muted">Please enter a path.</p>';
      return;
    }

    this.dom.provenanceResult.innerHTML = '<p class="muted">Tracing lineage...</p>';

    const data = await this._api(`/api/v1/lineage/${encodeURIComponent(path)}`);
    if (!data) {
      this.dom.provenanceResult.innerHTML = '<p class="error-msg">Could not retrieve lineage. The entry may not exist or the server returned an error.</p>';
      return;
    }

    const lineage = data.lineage || [];
    if (lineage.length === 0) {
      this.dom.provenanceResult.innerHTML = '<p class="muted">No lineage found for this path. It may be a root entry with no provenance chain.</p>';
      return;
    }

    const chain = document.createElement("div");
    chain.className = "lineage-chain";

    for (const node of lineage) {
      const div = document.createElement("div");
      div.className = "lineage-node";
      div.innerHTML = `
        <div class="lineage-connector">
          <div class="lineage-dot"></div>
          <div class="lineage-line"></div>
        </div>
        <span class="lineage-text">${this._esc(String(node))}</span>
      `;
      chain.appendChild(div);
    }

    this.dom.provenanceResult.innerHTML = "";
    this.dom.provenanceResult.appendChild(chain);
  }

  // ──────────────────────────────────────────────
  //  Tabs
  // ──────────────────────────────────────────────

  _initTabs() {
    const tabs = document.querySelectorAll(".tab");
    tabs.forEach(tab => {
      tab.addEventListener("click", () => this._switchTab(tab.dataset.tab));
    });
  }

  _switchTab(tabId) {
    // Deactivate all tabs
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.remove("active");
    });

    // Activate selected
    const tab = document.querySelector(`.tab[data-tab="${tabId}"]`);
    const panel = document.getElementById(`panel-${tabId}`);
    if (tab) {
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
    }
    if (panel) {
      panel.classList.add("active");
    }
  }

  // ──────────────────────────────────────────────
  //  Event bindings
  // ──────────────────────────────────────────────

  _bindEvents() {
    // Search filter
    this.dom.knowledgeSearch.addEventListener("input", () => {
      this._renderEntries(this.dom.knowledgeSearch.value);
    });

    // Clear events
    this.dom.clearEvents.addEventListener("click", () => {
      this.events = [];
      this.dom.eventFeed.innerHTML = '<p class="muted" id="eventPlaceholder">Waiting for events...</p>';
      this.dom.eventPlaceholder = document.getElementById("eventPlaceholder");
      this._updateEventCount();
    });

    // Provenance lookup
    this.dom.provenanceLookup.addEventListener("click", () => this._lookupProvenance());
    this.dom.provenancePath.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this._lookupProvenance();
    });

    // Sidebar toggle (mobile)
    this.dom.sidebarToggle.addEventListener("click", () => {
      this.dom.sidebar.classList.toggle("open");
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener("click", (e) => {
      if (
        this.dom.sidebar.classList.contains("open") &&
        !this.dom.sidebar.contains(e.target) &&
        !this.dom.sidebarToggle.contains(e.target)
      ) {
        this.dom.sidebar.classList.remove("open");
      }
    });
  }

  // ──────────────────────────────────────────────
  //  Utilities
  // ──────────────────────────────────────────────

  _esc(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  _truncate(str, max) {
    if (!str) return "";
    return str.length > max ? str.slice(0, max) + "\u2026" : str;
  }

  _formatTime(ts) {
    try {
      // ts could be ISO string or unix timestamp
      const date = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
      if (isNaN(date.getTime())) return String(ts);
      const now = new Date();
      const diffMs = now - date;
      if (diffMs < 60000) return "just now";
      if (diffMs < 3600000) return Math.floor(diffMs / 60000) + "m ago";
      if (diffMs < 86400000) return Math.floor(diffMs / 3600000) + "h ago";
      return date.toLocaleDateString();
    } catch {
      return String(ts);
    }
  }
}

// ──────────────────────────────────────────────
//  Boot
// ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new DashboardApp();
});
