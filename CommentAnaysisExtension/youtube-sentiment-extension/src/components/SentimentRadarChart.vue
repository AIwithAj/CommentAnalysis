<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="target"></i>
      Video Analysis Profile
    </div>
    <div class="chart-wrapper chart-small">
      <canvas ref="radarCanvas"></canvas>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';

export default {
  name: 'SentimentRadarChart',
  props: {
    analysisData: {
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
    analysisData: {
      handler() {
        this.updateChart();
      },
      deep: true
    }
  },
  methods: {
    initChart() {
      const ctx = this.$refs.radarCanvas.getContext('2d');
      
      const radarData = this.processRadarData();
      
      this.chart = new Chart(ctx, {
        type: 'radar',
        data: {
          labels: [
            'Positivity',
            'Engagement',
            'Word Diversity',
            'Activity Level',
            'Comment Quality',
            'Discussion Depth'
          ],
          datasets: [{
            label: 'Video Analysis',
            data: radarData,
            backgroundColor: 'rgba(0, 212, 255, 0.2)',
            borderColor: 'rgba(0, 212, 255, 1)',
            borderWidth: 2,
            pointBackgroundColor: 'rgba(0, 212, 255, 1)',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7
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
              cornerRadius: 10
            }
          },
          scales: {
            r: {
              beginAtZero: true,
              max: 10,
              ticks: {
                color: 'rgba(255, 255, 255, 0.7)',
                backdropColor: 'transparent',
                font: {
                  size: 10
                }
              },
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              angleLines: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              pointLabels: {
                color: 'rgba(255, 255, 255, 0.8)',
                font: {
                  size: 11,
                  weight: '500'
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
    processRadarData() {
      const metrics = this.analysisData.metrics;
      const sentiment = this.analysisData.sentiment_distribution;
      
      const positivity = ((sentiment.positive / (sentiment.positive + sentiment.neutral + sentiment.negative)) * 10);
      const engagement = Math.min(metrics.engagement_score / 10, 10);
      const wordDiversity = Math.min(metrics.avg_words_per_comment / 2, 10);
      const activityLevel = Math.min(Math.log10(metrics.total_comments + 1) * 2, 10);
      const commentQuality = metrics.sentiment_score;
      const discussionDepth = Math.min((metrics.unique_commenters / metrics.total_comments) * 20, 10);
      
      return [
        positivity,
        engagement,
        wordDiversity,
        activityLevel,
        commentQuality,
        discussionDepth
      ];
    },
    updateChart() {
      if (this.chart) {
        const radarData = this.processRadarData();
        this.chart.data.datasets[0].data = radarData;
        this.chart.update('active');
      }
    }
  }
};
</script>
