<template>
  <div class="glass-card chart-container">
    <div class="chart-title">
      <i data-feather="cloud"></i>
      Popular Keywords
    </div>
    <div class="wordcloud-container" ref="wordcloudContainer">
      <div 
        v-for="(word, index) in displayWords" 
        :key="index"
        :class="getWordClass(word[1])"
        :style="getWordStyle(word[1], index)"
        class="word-item"
        @click="highlightWord(word[0])"
      >
        {{ word[0] }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'WordCloud',
  props: {
    wordData: {
      type: Array,
      required: true
    }
  },
  computed: {
    displayWords() {
      return this.wordData.slice(0, 30);
    }
  },
  mounted() {
    this.$nextTick(() => {
      if (window.feather) {
        window.feather.replace();
      }
    });
  },
  methods: {
    getWordClass(frequency) {
      if (frequency > 10) return 'word-positive';
      if (frequency > 5) return 'word-neutral';
      return 'word-negative';
    },
    getWordStyle(frequency, index) {
      const maxFreq = this.wordData[0] ? this.wordData[0][1] : 1;
      const minSize = 12;
      const maxSize = 24;
      const fontSize = minSize + (frequency / maxFreq) * (maxSize - minSize);
      
      return {
        fontSize: `${fontSize}px`,
        animationDelay: `${index * 0.1}s`,
        transform: `rotate(${Math.random() * 20 - 10}deg)`
      };
    },
    highlightWord(word) {
      console.log('Highlighted word:', word);
    }
  }
};
</script>
