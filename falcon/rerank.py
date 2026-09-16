"""k-reciprocal re-ranking, совместимый с потоковым протоколом.

Классический алгоритм (Zhong et al., 2017) строит совместную матрицу расстояний
по объединению [все запросы; галерея] и использует в том числе связи «запрос —
запрос». Ответ 38 организаторов это запрещает: каждый query обрабатывается
независимо, как кадр из видеопотока, и знать о других запросах решение не может.

Здесь используется только:
  * близость «текущий запрос -> галерея»  (есть по условию);
  * близость «галерея -> галерея»          (галерея статична, строится заранее).
Связей «запрос -> другой запрос» нет вообще, поэтому выдача для запроса не
зависит от того, какие ещё запросы лежат в файле.

Идея метода: если A нашёл B среди соседей И B нашёл A среди своих, связь взаимна
и заслуживает доверия. Односторонние связи (похожий цвет, похожий ракурс) так
отсеиваются. Вместо чистого косинуса берётся расстояние Жаккара между
множествами взаимных соседей.

Тонкость для внешнего запроса: запрос не лежит в галерее, поэтому «попал ли
запрос в окрестность кандидата» проверяется сравнением с порогом — близостью
k1-го соседа этого кандидата внутри галереи.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import l2_normalize


@dataclass
class GalleryIndex:
    """Предрассчитанная структура соседства галереи.

    Строится один раз при загрузке базы и переиспользуется всеми запросами:
    галерея статична, поэтому это часть подготовки индекса, а не обработки запроса.
    """

    vectors: np.ndarray            # (N, D), L2-нормированные
    neighbours: np.ndarray         # (N, k) индексы соседей по убыванию близости
    similarities: np.ndarray       # (N, k) их близости
    k: int

    def threshold(self, index: int, k: int) -> float:
        """Близость k-го соседа: порог попадания в окрестность объекта."""
        return float(self.similarities[index, min(k, self.k) - 1])

    def reciprocal_set(self, index: int, k: int) -> set[int]:
        """Взаимные соседи объекта галереи."""
        k = min(k, self.k)
        result = {index}
        for rank in range(k):
            other = int(self.neighbours[index, rank])
            # Взаимность: index должен попадать в окрестность other.
            if self.similarities[index, rank] >= self.threshold(other, k):
                result.add(other)
        return result


def build_gallery_index(gallery_vectors: np.ndarray, k: int = 30) -> GalleryIndex:
    vectors = l2_normalize(gallery_vectors)
    size = len(vectors)
    k = max(1, min(k, size - 1))
    neighbours = np.zeros((size, k), dtype=np.int32)
    similarities = np.zeros((size, k), dtype=np.float32)

    for start in range(0, size, 512):
        block = vectors[start : start + 512] @ vectors.T
        for offset in range(len(block)):
            row = start + offset
            scores = block[offset].copy()
            scores[row] = -np.inf  # объект не сосед сам себе
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            neighbours[row] = top
            similarities[row] = scores[top]

    return GalleryIndex(vectors=vectors, neighbours=neighbours, similarities=similarities, k=k)


def _expand(base: set[int], index: GalleryIndex, k1: int, k2: int) -> set[int]:
    """Расширение окрестности: добавляем соседей соседей при прочной связи."""
    expanded = set(base)
    half = max(1, k1 // 2)
    for member in list(base)[:k2]:
        candidate = index.reciprocal_set(member, half)
        if len(candidate & base) >= len(candidate) * 2 / 3:
            expanded |= candidate
    return expanded


def rerank_query(
    query_vector: np.ndarray,
    index: GalleryIndex,
    *,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    candidate_pool: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Переранжирует кандидатов одного запроса.

    Возвращает (индексы галереи в новом порядке, оценки — больше значит увереннее).
    lambda_value смешивает исходный косинус и расстояние Жаккара: 1.0 — вернуться
    к исходному порядку, 0.0 — довериться только взаимному соседству.
    """
    query = l2_normalize(query_vector.reshape(1, -1))[0]
    similarity = index.vectors @ query

    pool_size = min(candidate_pool, len(index.vectors))
    pool = np.argpartition(-similarity, pool_size - 1)[:pool_size]
    pool = pool[np.argsort(-similarity[pool], kind="stable")]

    # Взаимная окрестность запроса: кандидат входит, если запрос в свою очередь
    # достаточно близок к нему — ближе, чем его собственный k1-й сосед.
    query_set: set[int] = set()
    for candidate in pool[:k1]:
        candidate = int(candidate)
        if similarity[candidate] >= index.threshold(candidate, k1):
            query_set.add(candidate)
    if not query_set:
        # Ни одной взаимной связи — переранжировать нечем, отдаём косинус как есть.
        order = np.argsort(-similarity, kind="stable")
        return order, similarity[order]
    query_set = _expand(query_set, index, k1, k2)

    jaccard = np.ones(len(index.vectors), dtype=np.float32)
    for candidate in pool:
        candidate = int(candidate)
        candidate_set = _expand(index.reciprocal_set(candidate, k1), index, k1, k2)
        union = len(query_set | candidate_set)
        if union:
            jaccard[candidate] = 1.0 - len(query_set & candidate_set) / union

    distance = (1.0 - similarity) / 2.0
    combined = lambda_value * distance + (1.0 - lambda_value) * jaccard
    order = np.argsort(combined, kind="stable")
    return order, -combined[order]
