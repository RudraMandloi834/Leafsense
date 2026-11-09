// Service Worker for AgriScan AI PWA
const CACHE_NAME = 'agriscan-v2' // Bumped version to force cache refresh
const urlsToCache = [
    '/',
    '/index.html',
]

// Install event - cache resources
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Opened cache')
                return cache.addAll(urlsToCache)
            })
            .catch((err) => {
                console.log('Cache install error:', err)
            })
    )
    self.skipWaiting()
})

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName)
                        return caches.delete(cacheName)
                    }
                })
            )
        })
    )
    self.clients.claim()
})

// Fetch event - Network first for CSS/JS, cache first for others
self.addEventListener('fetch', (event) => {
    // Skip cross-origin requests
    if (!event.request.url.startsWith(self.location.origin)) {
        return
    }

    const url = new URL(event.request.url)

    // Network-first strategy for CSS, JS, and API calls
    if (url.pathname.endsWith('.css') ||
        url.pathname.endsWith('.js') ||
        url.pathname.includes('/assets/') ||
        url.pathname.includes('/api/') ||
        url.hostname === '127.0.0.1') {

        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Clone and cache the response
                    const responseToCache = response.clone()
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseToCache)
                    })
                    return response
                })
                .catch(() => {
                    // Fallback to cache if network fails
                    return caches.match(event.request)
                })
        )
        return
    }

    // Cache-first strategy for other resources
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Cache hit - return response
                if (response) {
                    return response
                }

                // Clone the request
                const fetchRequest = event.request.clone()

                return fetch(fetchRequest).then((response) => {
                    // Check if valid response
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response
                    }

                    // Clone the response
                    const responseToCache = response.clone()

                    caches.open(CACHE_NAME)
                        .then((cache) => {
                            cache.put(event.request, responseToCache)
                        })

                    return response
                })
            })
            .catch(() => {
                // Return offline page if available
                return caches.match('/index.html')
            })
    )
})

// Background sync for offline predictions (future enhancement)
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-predictions') {
        event.waitUntil(syncPredictions())
    }
})

async function syncPredictions() {
    // Placeholder for syncing offline predictions when back online
    console.log('Syncing predictions...')
}
