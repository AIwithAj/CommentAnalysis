import { createApp } from 'vue';
import DashboardApp from './components/DashboardApp.vue';

// Create Vue application
createApp(DashboardApp).mount('#app');

// Initialize Feather icons when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  if (window.feather) {
    feather.replace();
  }
});
