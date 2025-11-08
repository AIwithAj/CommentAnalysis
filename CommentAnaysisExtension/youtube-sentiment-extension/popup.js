(function () {
  try {
    const params = new URLSearchParams(location.search);
    const videoUrl = params.get("url");
    if (videoUrl) {
      window.__LAUNCH_VIDEO_URL__ = videoUrl;
      console.log("[popup] Auto-launch with URL:", videoUrl);
    }
  } catch (e) {
    console.warn("Failed to parse ?url=", e);
  }
})();
