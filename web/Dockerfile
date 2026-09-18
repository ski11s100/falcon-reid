# ФАЛЬКОН — клиентская часть: интерфейс оператора и обратный прокси к API.
#
# Непривилегированный образ nginx: процесс работает не от root и слушает
# порт 8080, поэтому контейнеру не нужны никакие дополнительные права.
# Контекст сборки — корень репозитория: интерфейс лежит в service/static,
# откуда его же отдаёт сервис при запуске без Docker.

FROM nginxinc/nginx-unprivileged:1.29-alpine

COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY service/static /usr/share/nginx/html/static

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
