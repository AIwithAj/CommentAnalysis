<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="gauge"></i>
      Sentiment Gauge
    </div>
    <div class="gauge-container">
      <div class="gauge-chart">
        <canvas ref="gaugeCanvas"></canvas>
        <div class="gauge-center">
          <div class="gauge-value">{{ sentimentScore.toFixed(1) }}</div>
          <div class="gauge-label">{{ sentimentLabel }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';

export default {
  name: 'SentimentGaugeChart',
  props: {
    sentimentScore: {
      type: Number,
      required: true
    }
  },
  data() {
    return {
      chart: null
    };
  },
  mounted() {
    this.initChart();
    this.$nextTick(() => {
      if (window.feather) {
        window.feather.replace();
      }
    });
  },
  beforeUnmount() {
    if (this.chart) {
      this.chart.destroy();
    }
  },
  watch: {
    sentimentScore: {
      handler() {
        this.updateChart();
      }
    }
  },
  computed: {
    gaugeColor() {
      if (this.sentimentScore >= 7) return '#4facfe';
      if (this.sentimentScore >= 4) return '#667eea';
      return '#fa709a';
    },
    sentimentLabel() {
      if (this.sentimentScore >= 7) return 'Very Positive';
      if (this.sentimentScore >= 6) return 'Positive';
      if (this.sentimentScore >= 4) return 'Neutral';
      if (this.sentimentScore >= 3) return 'Negative';
      return 'Very Negative';
    }
  },
  methods: {
    initChart() {
      const ctx = this.$refs.gaugeCanvas.getContext('2d');
      
      this.chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          datasets: [{
            data: [this.sentimentScore, 10 - this.sentimentScore],
            backgroundColor: [
              this.gaugeColor,
              'rgba(255, 255, 255, 0.1)'
            ],
            borderWidth: 0,
            cutout: '80%',
            circumference: 180,
            rotation: 270
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              enabled: false
            }
          },
          animation: {
            animateRotate: true,
            duration: 2000,
            easing: 'easeOutQuart'
          }
        }
      });
    },
    updateChart() {
      if (this.chart) {
        this.chart.data.datasets[0].data = [this.sentimentScore, 10 - this.sentimentScore];
        this.chart.data.datasets[0].backgroundColor[0] = this.gaugeColor;
        this.chart.update('active');
      }
    }
  }
};
</script>
