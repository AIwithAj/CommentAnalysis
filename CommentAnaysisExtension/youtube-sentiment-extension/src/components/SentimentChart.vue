<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="pie-chart"></i>
      Sentiment Distribution
    </div>
    <div class="chart-wrapper chart-small">
      <canvas ref="chartCanvas"></canvas>
    </div>
    <div class="chart-legend">
      <div class="legend-item">
        <div class="legend-color bg-gradient-success"></div>
        <span>Positive ({{ sentimentData.positive }})</span>
      </div>
      <div class="legend-item">
        <div class="legend-color bg-gradient-secondary"></div>
        <span>Neutral ({{ sentimentData.neutral }})</span>
      </div>
      <div class="legend-item">
        <div class="legend-color bg-gradient-danger"></div>
        <span>Negative ({{ sentimentData.negative }})</span>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';

export default {
  name: 'SentimentChart',
  props: {
    sentimentData: {
      type: Object,
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
    sentimentData: {
      handler() {
        this.updateChart();
      },
      deep: true
    }
  },
  methods: {
    initChart() {
      const ctx = this.$refs.chartCanvas.getContext('2d');
      
      this.chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: ['Positive', 'Neutral', 'Negative'],
          datasets: [{
            data: [
              this.sentimentData.positive,
              this.sentimentData.neutral,
              this.sentimentData.negative
            ],
            backgroundColor: [
              'rgba(79, 172, 254, 0.8)',
              'rgba(102, 126, 234, 0.8)',
              'rgba(250, 112, 154, 0.8)'
            ],
            borderColor: [
              'rgba(79, 172, 254, 1)',
              'rgba(102, 126, 234, 1)',
              'rgba(250, 112, 154, 1)'
            ],
            borderWidth: 2,
            hoverOffset: 10
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
              backgroundColor: 'rgba(15, 15, 15, 0.9)',
              titleColor: '#ffffff',
              bodyColor: '#ffffff',
              borderColor: 'rgba(0, 212, 255, 0.5)',
              borderWidth: 1,
              cornerRadius: 10,
              displayColors: true,
              callbacks: {
                label: (context) => {
                  const total = context.dataset.data.reduce((a, b) => a + b, 0);
                  const percentage = ((context.parsed / total) * 100).toFixed(1);
                  return `${context.label}: ${context.parsed} (${percentage}%)`;
                }
              }
            }
          },
          cutout: '60%',
          animation: {
            animateRotate: true,
            animateScale: true,
            duration: 1500,
            easing: 'easeOutQuart'
          },
          elements: {
            arc: {
              borderRadius: 8
            }
          }
        }
      });
    },
    updateChart() {
      if (this.chart) {
        this.chart.data.datasets[0].data = [
          this.sentimentData.positive,
          this.sentimentData.neutral,
          this.sentimentData.negative
        ];
        this.chart.update('active');
      }
    }
  }
};
</script>
