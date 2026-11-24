document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggleBtn");
  const whitelistBtn = document.getElementById("whitelistBtn");
  const serverBadge = document.getElementById("serverBadge");

  const statReqs = document.getElementById("statReqs");
  const statHits = document.getElementById("statHits");
  const statMs = document.getElementById("statMs");

  const checkIds = [
    "streamer_safe_mode",
    "blur_face",
    "blur_hair",
    "blur_chest",
    "blur_limbs",
    "enable_anime_games",
    "blur_fullbody",
    "show_hud"
  ];

  const modeSelect = document.getElementById("blur_mode");
  const sliderIntensity = document.getElementById("blur_intensity");
  const sliderPadding = document.getElementById("padding_pct");
  const sliderSens = document.getElementById("sens_pct");

  const valIntensity = document.getElementById("val_intensity");
  const valPadding = document.getElementById("val_padding");
  const valSens = document.getElementById("val_sens");

  let isEnabled = true;
  let currentHost = "";
  let whitelistedHosts = [];

  function updateToggleUI() {
    if (isEnabled) {
      toggleBtn.textContent = "🟢 البلور شغّال ويغطي الحين (اضغط لا بغيت توقفه)";
      toggleBtn.classList.remove("off");
    } else {
      toggleBtn.textContent = "🔴 البلور طافي الحين (اضغط عشان تشغّله)";
      toggleBtn.classList.add("off");
    }
  }

  function updateWhitelistUI() {
    if (!currentHost) {
      whitelistBtn.style.display = "none";
      return;
    }
    whitelistBtn.style.display = "block";
    if (whitelistedHosts.includes(currentHost)) {
      whitelistBtn.textContent = `✅ هالموقع مسموح وما عليه بلور (${currentHost}) — اضغط لإلغاء الاستثناء`;
      whitelistBtn.classList.add("active");
    } else {
      whitelistBtn.textContent = `⭐ خلّ هالموقع مسموح ولا تغبّش فيه (${currentHost})`;
      whitelistBtn.classList.remove("active");
    }
  }

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs && tabs[0] && tabs[0].url) {
      try {
        const u = new URL(tabs[0].url);
        if (u.protocol.startsWith("http")) {
          currentHost = u.hostname;
        }
      } catch (e) {}
    }
    updateWhitelistUI();
  });

  function applyStateToUI(res) {
    if (res.enabled !== undefined) isEnabled = Boolean(res.enabled);
    if (Array.isArray(res.whitelisted_hosts)) {
      whitelistedHosts = res.whitelisted_hosts;
    }
    updateToggleUI();
    updateWhitelistUI();

    for (const id of checkIds) {
      if (res[id] !== undefined) {
        const el = document.getElementById(id);
        if (el) el.checked = Boolean(res[id]);
      }
    }

    if (res.blur_mode) modeSelect.value = res.blur_mode;
    if (res.blur_intensity !== undefined) {
      sliderIntensity.value = res.blur_intensity;
      valIntensity.textContent = `${res.blur_intensity}px`;
    }
    if (res.padding_pct !== undefined) {
      sliderPadding.value = res.padding_pct;
      valPadding.textContent = `${res.padding_pct}%`;
    }
    if (res.sens_pct !== undefined) {
      sliderSens.value = res.sens_pct;
      valSens.textContent = `${res.sens_pct}%`;
    }
  }

  chrome.storage.local.get(
    [
      "enabled",
      "whitelisted_hosts",
      "blur_mode",
      "blur_intensity",
      "padding_pct",
      "sens_pct",
      ...checkIds
    ],
    (res) => applyStateToUI(res)
  );

  function refreshServerStats() {
    chrome.runtime.sendMessage({ type: "CHECK_SERVER" }, (resp) => {
      if (resp && resp.online) {
        serverBadge.textContent = `⚡ شغّال ومشبوك مع كرت الشاشة (${resp.hardware})`;
        serverBadge.style.color = "#34D399";
        if (statReqs) statReqs.textContent = resp.total_requests || 0;
        if (statHits) statHits.textContent = resp.cache_hits || 0;
        if (statMs) statMs.textContent = resp.avg_ms ? `${resp.avg_ms}ms` : "سريع";
      } else {
        serverBadge.textContent = "⚠️ السيرفر طافي — شغّل البرنامج عشان يشتغل الذكاء الاصطناعي";
        serverBadge.style.color = "#FBBF24";
      }
    });
  }
  refreshServerStats();
  setInterval(refreshServerStats, 1500);

  toggleBtn.addEventListener("click", () => {
    isEnabled = !isEnabled;
    updateToggleUI();
    chrome.storage.local.set({ enabled: isEnabled });
  });

  whitelistBtn.addEventListener("click", () => {
    if (!currentHost) return;
    if (whitelistedHosts.includes(currentHost)) {
      whitelistedHosts = whitelistedHosts.filter((h) => h !== currentHost);
    } else {
      whitelistedHosts.push(currentHost);
    }
    updateWhitelistUI();
    chrome.storage.local.set({ whitelisted_hosts: whitelistedHosts });
  });

  document.getElementById("presetStreamer").addEventListener("click", () => {
    const preset = {
      enabled: true,
      streamer_safe_mode: true,
      blur_face: true,
      blur_hair: true,
      blur_chest: true,
      blur_limbs: true,
      enable_anime_games: true,
      blur_fullbody: false,
      blur_intensity: 32,
      padding_pct: 14,
      sens_pct: 80
    };
    applyStateToUI(preset);
    chrome.storage.local.set(preset);
  });

  document.getElementById("presetBalanced").addEventListener("click", () => {
    const preset = {
      enabled: true,
      streamer_safe_mode: false,
      blur_face: true,
      blur_hair: true,
      blur_chest: true,
      blur_limbs: true,
      enable_anime_games: true,
      blur_fullbody: false,
      blur_intensity: 26,
      padding_pct: 10,
      sens_pct: 75
    };
    applyStateToUI(preset);
    chrome.storage.local.set(preset);
  });

  document.getElementById("presetFullGirl").addEventListener("click", () => {
    const preset = {
      enabled: true,
      streamer_safe_mode: true,
      blur_face: true,
      blur_hair: true,
      blur_chest: true,
      blur_limbs: true,
      enable_anime_games: true,
      blur_fullbody: true,
      blur_intensity: 34,
      padding_pct: 15,
      sens_pct: 78
    };
    applyStateToUI(preset);
    chrome.storage.local.set(preset);
  });

  for (const id of checkIds) {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", (e) => {
        chrome.storage.local.set({ [id]: e.target.checked });
      });
    }
  }

  modeSelect.addEventListener("change", (e) => {
    chrome.storage.local.set({ blur_mode: e.target.value });
  });

  sliderIntensity.addEventListener("input", (e) => {
    valIntensity.textContent = `${e.target.value}px`;
    chrome.storage.local.set({ blur_intensity: Number(e.target.value) });
  });

  sliderPadding.addEventListener("input", (e) => {
    valPadding.textContent = `${e.target.value}%`;
    chrome.storage.local.set({ padding_pct: Number(e.target.value) });
  });

  sliderSens.addEventListener("input", (e) => {
    valSens.textContent = `${e.target.value}%`;
    chrome.storage.local.set({ sens_pct: Number(e.target.value) });
  });
});
