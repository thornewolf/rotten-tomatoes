/**
 * Signal Widget Component for Alpine.js
 *
 * Handles fetching and displaying ML model signals for Kalshi markets.
 * Provides inputs for RT features and displays prediction results.
 */

document.addEventListener('alpine:init', () => {
    Alpine.data('signalWidget', (ticker) => ({
        ticker,
        days_since_release: '',
        current_rating: '',
        num_reviews: '',
        model: 'default',
        loading: false,
        error: '',
        result: null,

        /**
         * Fetch signal from the API with the current feature values.
         */
        async run() {
            this.loading = true;
            this.error = '';
            this.result = null;

            try {
                const params = new URLSearchParams();

                // Only include non-empty parameters
                if (this.days_since_release !== '') {
                    params.append('days_since_release', this.days_since_release);
                }
                if (this.current_rating !== '') {
                    params.append('current_rating', this.current_rating);
                }
                if (this.num_reviews !== '') {
                    params.append('num_reviews', this.num_reviews);
                }
                if (this.model && this.model !== 'default') {
                    params.append('model', this.model);
                }

                const qs = params.toString();
                const url = qs
                    ? `/api/markets/${this.ticker}/signals?${qs}`
                    : `/api/markets/${this.ticker}/signals`;

                const resp = await fetch(url);
                if (!resp.ok) {
                    throw new Error(`HTTP ${resp.status}`);
                }

                this.result = await resp.json();
            } catch (err) {
                this.error = err?.message || 'Failed to fetch signal';
            } finally {
                this.loading = false;
            }
        },
    }));
});
