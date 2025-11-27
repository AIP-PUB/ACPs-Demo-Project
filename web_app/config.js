(function setRuntimeConfig() {
    const defaults = {
        backendBase: 'http://127.0.0.1:8019',
    };
    window.APP_CONFIG = Object.assign({}, defaults, window.APP_CONFIG || {});
})();
