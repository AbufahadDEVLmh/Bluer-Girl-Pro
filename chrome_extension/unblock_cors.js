(() => {
  if (window.__bluerGirlCorsHooked) return;
  window.__bluerGirlCorsHooked = true;

  function shouldApplyAnonymousCors(urlStr) {
    if (!urlStr || typeof urlStr !== "string") return false;
    if (urlStr.startsWith("blob:") || urlStr.startsWith("data:")) return false;
    try {
      const u = new URL(urlStr, window.location.href);
      return u.origin !== window.location.origin;
    } catch (e) {
      return false;
    }
  }

  function attachFallbackOnError(el) {
    if (el.__bgErrorHooked) return;
    el.__bgErrorHooked = true;
    el.addEventListener(
      "error",
      () => {
        if (el.crossOrigin) {
          const savedSrc = el.getAttribute("src") || el.src;
          el.removeAttribute("crossorigin");
          el.__bgSkipCors = true;
          if (savedSrc) {
            el.src = savedSrc;
            if (el.tagName === "VIDEO") {
              try {
                el.load();
                el.play().catch(() => {});
              } catch (e) {}
            }
          }
        }
      },
      { capture: true }
    );
  }

  function prepareMediaElement(el) {
    if (!el || el.__bgSkipCors) return;
    attachFallbackOnError(el);
    const src = el.getAttribute("src") || el.src;
    if (shouldApplyAnonymousCors(src) && !el.crossOrigin) {
      if (el.tagName === "IMG" && !el.complete) {
        el.crossOrigin = "anonymous";
      } else if (el.tagName === "VIDEO" && el.readyState === 0) {
        el.crossOrigin = "anonymous";
      }
    }
  }

  // Hook setAttribute on HTMLVideoElement and HTMLImageElement for direct cross-origin URLs
  const origSetAttr = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    if (
      (this.tagName === "VIDEO" || this.tagName === "IMG") &&
      !this.__bgSkipCors &&
      typeof name === "string" &&
      name.toLowerCase() === "src"
    ) {
      attachFallbackOnError(this);
      if (shouldApplyAnonymousCors(String(value))) {
        origSetAttr.call(this, "crossorigin", "anonymous");
      }
    }
    return origSetAttr.call(this, name, value);
  };

  // Hook .src property setter on HTMLMediaElement & HTMLImageElement
  for (const Proto of [HTMLMediaElement.prototype, HTMLImageElement.prototype]) {
    const desc = Object.getOwnPropertyDescriptor(Proto, "src");
    if (desc && desc.set) {
      Object.defineProperty(Proto, "src", {
        ...desc,
        set(val) {
          if (!this.__bgSkipCors) {
            attachFallbackOnError(this);
            if (shouldApplyAnonymousCors(String(val))) {
              this.crossOrigin = "anonymous";
            }
          }
          return desc.set.call(this, val);
        }
      });
    }
  }

  // Observe newly added DOM nodes early
  const obs = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.tagName === "VIDEO" || node.tagName === "IMG") {
          prepareMediaElement(node);
        } else if (node.querySelectorAll) {
          node.querySelectorAll("video, img").forEach(prepareMediaElement);
        }
      }
    }
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();
