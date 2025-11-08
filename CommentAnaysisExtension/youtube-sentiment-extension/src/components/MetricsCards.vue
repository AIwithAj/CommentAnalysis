<template>
  <div class="metrics-grid slide-in-up">
    <div class="glass-card metric-card">
      <div class="metric-icon">
        <i data-feather="message-circle"></i>
      </div>
      <div class="metric-value">{{ formatNumber(metrics.total_comments) }}</div>
      <div class="metric-label">Total Comments</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon">
        <i data-feather="users"></i>
      </div>
      <div class="metric-value">{{ formatNumber(metrics.unique_commenters) }}</div>
      <div class="metric-label">Unique Users</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon">
        <i data-feather="trending-up"></i>
      </div>
      <div class="metric-value">{{ metrics.sentiment_score.toFixed(1) }}/10</div>
      <div class="metric-label">Sentiment Score</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon">
        <i data-feather="activity"></i>
      </div>
      <div class="metric-value">{{ metrics.engagement_score }}%</div>
      <div class="metric-label">Engagement</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon bg-gradient-success">
        <i data-feather="smile"></i>
      </div>
      <div class="metric-value">{{ positivePercentage }}%</div>
      <div class="metric-label">Positive</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon bg-gradient-secondary">
        <i data-feather="meh"></i>
      </div>
      <div class="metric-value">{{ neutralPercentage }}%</div>
      <div class="metric-label">Neutral</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon bg-gradient-danger">
        <i data-feather="frown"></i>
      </div>
      <div class="metric-value">{{ negativePercentage }}%</div>
      <div class="metric-label">Negative</div>
    </div>

    <div class="glass-card metric-card">
      <div class="metric-icon">
        <i data-feather="edit-3"></i>
      </div>
      <div class="metric-value">{{ metrics.avg_words_per_comment }}</div>
      <div class="metric-label">Avg Words</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MetricsCards',
  props: {
    metrics: {
      type: Object,
      required: true
    },
    sentiment: {
      type: Object,
      required: true
    }
  },
  computed: {
    totalComments() {
      return this.sentiment.positive + this.sentiment.neutral + this.sentiment.negative;
    },
    positivePercentage() {
      return Math.round((this.sentiment.positive / this.totalComments) * 100);
    },
    neutralPercentage() {
      return Math.round((this.sentiment.neutral / this.totalComments) * 100);
    },
    negativePercentage() {
      return Math.round((this.sentiment.negative / this.totalComments) * 100);
    }
  },
  methods: {
    formatNumber(num) {
      if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
      }
      return num.toString();
    }
  },
  mounted() {
    this.$nextTick(() => {
      if (window.feather) {
        window.feather.replace();
      }
    });
  },
  updated() {
    this.$nextTick(() => {
      if (window.feather) {
        window.feather.replace();
      }
    });
  }
};
</script>
