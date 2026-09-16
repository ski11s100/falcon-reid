import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from app.dataset import Observation
from scripts.export_official import write_rankings


class OfficialExportTests(unittest.TestCase):
    def test_gallery_only_order_and_refusal(self):
        def row(i):
            return Observation(i,str(i),Path(str(i)),(0,0,5,5),None,None)
        query, gallery = [row(99)], [row(i) for i in range(10)]
        vectors = np.array([[1,0]]+[[1,i/10] for i in range(10)],dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_rankings(folder,query,gallery,vectors,1.01)
            with (folder/'submission.csv').open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0],['query_id']+[f'gallery_id_{i}' for i in range(1,11)])
            self.assertEqual(rows[1],['99']+[str(i) for i in range(10)])
            with (folder/'candidates.csv').open() as f:
                self.assertEqual(list(csv.reader(f))[1],['99','',''])
