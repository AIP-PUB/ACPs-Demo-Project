(function setRuntimeConfig() {
    const defaults = {
        backendBase: 'http://127.0.0.1:59210', // 后端 Leader API 服务地址
        apiVersion: 'v1',            // API版本
        pollInterval: 5000,          // 轮询间隔（毫秒）
        maxPollRetries: 60,          // 最大轮询次数
    };
    window.APP_CONFIG = Object.assign({}, defaults, window.APP_CONFIG || {});
})();
