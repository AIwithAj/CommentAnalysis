<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="trending-up"></i>
      Sentiment Trends
    </div>
    <div class="chart-wrapper chart-small">
      <canvas ref="trendCanvas"></canvas>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';

export default {
  name: 'TrendChart',
  props: {
    trendData: {
      type: Array,
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
    trendData: {
      handler() {
        this.updateChart();
      },
      deep: true
    }
  },
  methods: {
    initChart() {
      const ctx = this.$refs.trendCanvas.getContext('2d');
      
      const processedData = this.processTrendData();
      
      this.chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: processedData.labels,
          datasets: [
            {
              label: 'Positive',
              data: processedData.positive,
              borderColor: 'rgba(79, 172, 254, 1)',
              backgroundColor: 'rgba(79, 172, 254, 0.1)',
              fill: true,
              tension: 0.4,
              pointRadius: 4,
              pointHoverRadius: 6,
              pointBackgroundColor: 'rgba(79, 172, 254, 1)',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 2
            },
            {
              label: 'Neutral',
              data: processedData.neutral,
              borderColor: 'rgba(102, 126, 234, 1)',
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
              fill: true,
              tension: 0.4,
              pointRadius: 4,
              pointHoverRadius: 6,
              pointBackgroundColor: 'rgba(102, 126, 234, 1)',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 2
            },
            {
              label: 'Negative',
              data: processedData.negative,
              borderColor: 'rgba(250, 112, 154, 1)',
              backgroundColor: 'rgba(250, 112, 154, 0.1)',
              fill: true,
              tension: 0.4,
              pointRadius: 4,
              pointHoverRadius: 6,
              pointBackgroundColor: 'rgba(250, 112, 154, 1)',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            intersect: false,
            mode: 'index'
          },
          plugins: {
            legend: {
              display: true,
              position: 'bottom',
              labels: {
                color: '#ffffff',
                usePointStyle: true,
                padding: 15,
                font: {
                  size: 11
                }
              }
            },
            tooltip: {
              backgroundColor: 'rgba(15, 15, 15, 0.9)',
              titleColor: '#ffffff',
              bodyColor: '#ffffff',
              borderColor: 'rgba(0, 212, 255, 0.5)',
              borderWidth: 1,
              cornerRadius: 10,
              displayColors: true
            }
          },
          scales: {
            x: {
              display: true,
              grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                drawBorder: false
              },
              ticks: {
                color: 'rgba(255, 255, 255, 0.7)',
                font: {
                  size: 10
                }
              }
            },
            y: {
              display: true,
              grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                drawBorder: false
              },
              ticks: {
                color: 'rgba(255, 255, 255, 0.7)',
                font: {
                  size: 10
                }
              }
            }
          },
          animation: {
            duration: 1500,
            easing: 'easeOutQuart'
          }
        }
      });
    },
    processTrendData() {
      if (!this.trendData || this.trendData.length === 0) {
        return {
          labels: [],
          positive: [],
          neutral: [],
          negative: []
        };
      }

      const labels = [];
      const positive = [];
      const neutral = [];
      const negative = [];

      this.trendData.forEach(item => {
        const date = new Date(item.hour);
        labels.push(date.toLocaleDateString() + ' ' + date.getHours() + ':00');
        positive.push(item.positive || 0);
        neutral.push(item.neutral || 0);
        negative.push(item.negative || 0);
      });

      return { labels, positive, neutral, negative };
    },
    updateChart() {
      if (this.chart) {
        const processedData = this.processTrendData();
        this.chart.data.labels = processedData.labels;
        this.chart.data.datasets[0].data = processedData.positive;
        this.chart.data.datasets[1].data = processedData.neutral;
        this.chart.data.datasets[2].data = processedData.negative;
        this.chart.update('active');
      }
    }
  }
};
</script>
