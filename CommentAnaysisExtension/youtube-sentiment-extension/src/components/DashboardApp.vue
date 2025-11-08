<template>
  <div class="dashboard-container">
    <!-- Header -->
    <div class="dashboard-header slide-in-up">


      <h1 class="dashboard-title flex items-center justify-center gap-2">
        <i data-feather="trending-up"></i>
        YouTube Sentiment Analysis Pro
      </h1>
      <div v-if="videoTitle" class="video-title text-center text-xl font-semibold mt-4">
        🎬 {{ videoTitle }}
      </div>
      <p class="dashboard-subtitle">
        Advanced AI-powered comment analysis with real-time insights
      </p>
      <div v-if="demoMode" class="demo-banner">
        <i data-feather="eye"></i>
        Demo Mode - Showing sample analysis
      </div>
    </div>

    <!-- URL Input & Launch Button -->
    <div v-if="!analysisData && !autoDetected" class="url-input-section glass-card">
      <input v-model="youtubeUrl" type="text" placeholder="Paste YouTube video URL here..." class="url-input" />
      <div class="toggle-wrapper">
        <label class="toggle-label">
          <input type="checkbox" v-model="quizo" />
          <span class="toggle-slider"></span>
          <span class="toggle-text">Enable Quiz Mode</span>
        </label>
      </div>
      <div class="button-group">
        <button class="btn btn-launch" @click="handleLaunch">
          <i data-feather="play"></i>
          Launch Analysis
        </button>
        <button class="btn btn-demo" @click="loadDemoData">
          <i data-feather="eye"></i>
          Demo
        </button>
      </div>
    </div>

    <!-- Charts Section -->
    <div v-if="analysisData" class="space-y-6">
      <MetricsCards :metrics="analysisData.metrics" :sentiment="analysisData.sentiment_distribution" />
      <SentimentChart :sentiment-data="analysisData.sentiment_distribution" />
      <SentimentGaugeChart :sentiment-score="analysisData.metrics.sentiment_score" />
      <TrendChart :trend-data="analysisData.trend_data" />
      <SentimentRadarChart :analysis-data="analysisData" />
      <WordCloud :word-data="analysisData.word_cloud_data" />
      <CommentsSection :comments="analysisData.top_comments" />
    </div>

    <!-- Loading -->
    <div v-if="loading" class="glass-card loading-container">
      <div class="loading-spinner"></div>
      <div class="loading-text">{{ loadingMessage }}</div>
      <div class="loading-subtext">Please wait while we analyze the comments...</div>
    </div>

    <!-- Error -->
    <div v-if="error" class="glass-card error-container">
      <i data-feather="alert-circle" class="error-icon"></i>
      <div class="error-message">{{ error }}</div>
      <div class="error-details">Please try refreshing the page or check your connection.</div>
    </div>
  </div>
</template>

<script>
import feather from 'feather-icons';
import MetricsCards from './MetricsCards.vue';
import SentimentChart from './SentimentChart.vue';
import SentimentGaugeChart from './SentimentGaugeChart.vue';
import TrendChart from './TrendChart.vue';
import SentimentRadarChart from './SentimentRadarChart.vue';
import WordCloud from './WordCloud.vue';
import CommentsSection from './CommentsSection.vue';

export default {
  name: 'DashboardApp',
  components: {
    MetricsCards,
    SentimentChart,
    SentimentGaugeChart,
    TrendChart,
    SentimentRadarChart,
    WordCloud,
    CommentsSection
  },
  data() {
    return {
      loading: false,
      error: null,
      loadingMessage: 'Analyzing YouTube comments...',
      demoMode: false,
      apiUrl: 'http://localhost:8000',
      analysisData: null,
      youtubeUrl: '',
      autoDetected: false,
      videoTitle: '',
      quizo: false
    };
  },
  async mounted() {
  feather.replace();

  // ✅ Auto-launch if popup.html?url=... was opened
  if (window.__LAUNCH_VIDEO_URL__) {
    console.log("[DashboardApp] Auto-launch with:", window.__LAUNCH_VIDEO_URL__);
    this.youtubeUrl = window.__LAUNCH_VIDEO_URL__;
    this.quizo = true; // or false if you want normal YouTube mode
    this.handleLaunch();
    return; // stop here
  }

  // Existing auto-detect logic (active YouTube tab)
  try {
    const tabs = await new Promise(resolve => {
      if (typeof chrome !== 'undefined' && chrome.tabs) {
        chrome.tabs.query({ active: true, currentWindow: true }, resolve);
      } else {
        resolve([]);
      }
    });

    const url = tabs[0]?.url || '';
    const videoId = this.extractVideoId(url);
    if (videoId) {
      this.autoDetected = true;
      await this.analyzeVideo(videoId);
    }
  } catch (e) {
    console.warn('Could not auto-detect YouTube video:', e);
  }
}
,
  methods: {
    extractVideoId(url) {
      const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);
      return match ? match[1] : null;
    },
    async handleLaunch() {
      this.loading = true;
      this.error = null;
      this.analysisData = null;

      try {
        if (this.quizo) {
          this.loadingMessage = 'Analyzing quiz feedback...';

          const res = await fetch(this.youtubeUrl);
          if (!res.ok) throw new Error(`Failed to fetch quiz feedback (HTTP ${res.status})`);

          const data = await res.json();
          const rawComments = Object.values(data).flat();  // ✅ Fix here

          if (!Array.isArray(rawComments)) {
            throw new Error('Expected an array of comments from the quiz feedback API');
          }

          const comments = rawComments.map(c => ({
            text: c.comment,
            timestamp: c.timestamp || new Date().toISOString(),
            authorId: `${c.user_id || 'Unknown'} ${c.username || ''}`.trim()
          }));

          const analysisRes = await fetch(`${this.apiUrl}/analyze_comments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comments })
          });

          if (!analysisRes.ok) throw new Error(`Failed to analyze feedback (HTTP ${analysisRes.status})`);

          this.analysisData = await analysisRes.json();
          this.videoTitle = 'Quiz Feedback Analysis';

        } else {
          const videoId = this.extractVideoId(this.youtubeUrl);

          if (!videoId) {
            this.error = 'Invalid YouTube URL';
            this.demoMode = true;
            await this.loadDemoData();
            return;
          }

          this.loadingMessage = 'Analyzing YouTube video...';
          await this.analyzeVideo(videoId);
        }
      } catch (e) {
        console.error(e);
        this.error = 'Failed to analyze video or feedback';
      } finally {
        this.loading = false;
      }
    }
    ,

    async loadDemoData() {
      this.loading = true;
      this.error = null;
      this.loadingMessage = 'Loading demo analysis...';
      try {
        const res = await fetch(`${this.apiUrl}/demo`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.analysisData = await res.json();
        this.demoMode = true;
      } catch (e) {
        this.error = 'Failed to load demo data';
      } finally {
        this.loading = false;
      }
    },

    async fetchComments(videoId) {
      let comments = [];
      let pageToken = '';
      const API_KEY = 'AIzaSyDbUL8ATWt5XCkTu-8rsHZZ5ItY8C5jBI8';

      try {
        while (comments.length < 500) {
          const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
          const data = await response.json();

          if (data.items) {
            data.items.forEach(item => {
              const snippet = item.snippet.topLevelComment.snippet;
              comments.push({
                text: snippet.textOriginal,
                timestamp: snippet.publishedAt,
                authorId: snippet.authorChannelId?.value || 'Unknown'
              });
            });
          }
          pageToken = data.nextPageToken;
          if (!pageToken) break;
        }
      } catch (error) {
        console.error('Error fetching comments:', error);
      }
      return comments;
    },

    async analyzeVideo(videoId) {
      const API_KEY = 'AIzaSyDbUL8ATWt5XCkTu-8rsHZZ5ItY8C5jBI8';
      this.loading = true;
      this.loadingMessage = 'Analyzing video...';
      try {
        const videoInfoRes = await fetch(`https://www.googleapis.com/youtube/v3/videos?part=snippet&id=${videoId}&key=${API_KEY}`);
        const videoInfoData = await videoInfoRes.json();
        this.videoTitle = videoInfoData.items?.[0]?.snippet?.title || '';
        const comments = await this.fetchComments(videoId);
        const res = await fetch(`${this.apiUrl}/analyze_comments`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comments })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.analysisData = await res.json();
        this.error = null;
      } catch (e) {
        this.error = 'Failed to analyze video';
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

<style scoped>
.space-y-6>*+* {
  margin-top: 1.5rem;
}

.flex {
  display: flex;
}

.items-center {
  align-items: center;
}

.justify-center {
  justify-content: center;
}

.gap-2 {
  gap: 0.5rem;
}

.url-input-section {
  margin-top: 1rem;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.url-input {
  width: 100%;
  max-width: 500px;
  padding: 0.75rem 1rem;
  border: 1px solid #ccc;
  border-radius: 0.5rem;
  font-size: 1rem;
}

.button-group {
  display: flex;
  gap: 1rem;
}

.btn {
  padding: 0.5rem 1.5rem;
  font-weight: 600;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease;
}

.btn-launch {
  background: linear-gradient(to right, #4facfe, #00f2fe);
  color: white;
}

.btn-demo {
  background: linear-gradient(to right, #43e97b, #38f9d7);
  color: white;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.toggle-wrapper {
  margin: 10px 0;
}

.toggle-label {
  display: flex;
  align-items: center;
  cursor: pointer;
}

.toggle-label input[type="checkbox"] {
  display: none;
}

.toggle-slider {
  width: 40px;
  height: 20px;
  background: #ccc;
  border-radius: 15px;
  margin-right: 10px;
  position: relative;
  transition: background 0.3s;
}

.toggle-slider::before {
  content: "";
  position: absolute;
  width: 18px;
  height: 18px;
  background: white;
  border-radius: 50%;
  top: 1px;
  left: 1px;
  transition: transform 0.3s;
}

.toggle-label input[type="checkbox"]:checked+.toggle-slider {
  background: #4CAF50;
}

.toggle-label input[type="checkbox"]:checked+.toggle-slider::before {
  transform: translateX(20px);
}

.toggle-text {
  font-size: 14px;
  color: #555;
}
</style>
