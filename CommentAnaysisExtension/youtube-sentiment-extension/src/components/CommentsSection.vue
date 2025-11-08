<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="message-square"></i>
      Comment Analysis
    </div>
    
    <!-- Tab Navigation -->
    <div class="nav-tabs">
      <button 
        v-for="tab in tabs" 
        :key="tab.id"
        :class="['nav-link', { active: activeTab === tab.id }]"
        @click="setActiveTab(tab.id)"
      >
        {{ tab.label }} ({{ tab.count }})
      </button>
    </div>
    
    <!-- Comments List -->
    <div v-for="comment in displayedComments" :key="comment.id" class="comment-item">
  <div class="comment-text">{{ comment.comment }}</div>
  <div class="comment-meta">
    <span class="flex items-center gap-1">
      <i data-feather="clock" class="w-3 h-3"></i>
      {{ comment.timestamp }}
    </span>
    <span :class="['sentiment-badge', `sentiment-${comment.sentiment_label}`]">
      {{ comment.sentiment_label }}
    </span>
  </div>
</div>

    
    <!-- Load More Button -->
    <div v-if="hasMore" class="text-center mt-4">
      <button 
        class="btn btn-primary"
        @click="loadMore"
      >
        <i data-feather="chevron-down" class="w-4 h-4"></i>
        Show More
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CommentsSection',
  props: {
    comments: {
      type: Object,
      required: true,
      default: () => ({
        positive: [],
        negative: [],
        most_engaged: []
      })
    }
  },
  data() {
    return {
      activeTab: 'all',
      displayLimit: 15,
      hasMore: true
    };
  },
  computed: {
    tabs() {
      const pos = this.comments.positive || [];
      const neg = this.comments.negative || [];
      const eng = this.comments.most_engaged || [];

      return [
        { id: 'all', label: 'All Comments', count: pos.length + neg.length },
        { id: 'positive', label: 'Positive', count: pos.length },
        { id: 'negative', label: 'Negative', count: neg.length },
        { id: 'engaged', label: 'Most Engaged', count: eng.length }
      ];
    },
    filteredComments() {
      switch (this.activeTab) {
        case 'positive':
          return this.comments.positive || [];
        case 'negative':
          return this.comments.negative || [];
        case 'engaged':
          return this.comments.most_engaged || [];
        case 'all':
        default:
          return [...(this.comments.positive || []), ...(this.comments.negative || [])];
      }
    },
    displayedComments() {
      return this.filteredComments.slice(0, this.displayLimit);
    }
  },
  watch: {
    activeTab() {
      this.displayLimit = 5;
      this.updateHasMore();
    },
    displayLimit() {
      this.updateHasMore();
    }
  },
  mounted() {
    this.updateHasMore();
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
  },
  methods: {
    setActiveTab(tabId) {
      this.activeTab = tabId;
    },
    loadMore() {
      this.displayLimit += 5;
    },
    updateHasMore() {
      this.hasMore = this.displayedComments.length > this.displayLimit;
    }
  }
};
</script>


<style scoped>
.flex {
  display: flex;
}

.items-center {
  align-items: center;
}

.gap-1 {
  gap: 0.25rem;
}

.w-3 {
  width: 0.75rem;
}

.h-3 {
  height: 0.75rem;
}

.w-4 {
  width: 1rem;
}

.h-4 {
  height: 1rem;
}

.mt-4 {
  margin-top: 1rem;
}

.text-center {
  text-align: center;
}

.btn {
  padding: 0.5rem 1.5rem;
  border: none;
  border-radius: 0.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.btn-primary {
  background: var(--primary-gradient);
  color: white;
}

.btn-primary:hover {
  box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
  transform: translateY(-2px);
}
</style>
