import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from app.dataset import read_manifest, audit_manifest, identity_split, grouped_identity_split, Observation
from app.fingerprint import crop_bbox, InvalidImage
from app.retrieval import evaluate, ranked, calibrate, normalize


class ChallengeTests(unittest.TestCase):
    def test_identical_frames_keep_different_identities_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for i,label in enumerate(['a','b','c','d']):
                path=root/f'{i}.png'
                Image.new('RGB',(20,20),(10 if i<2 else 30*i,0,0)).save(path)
                rows.append(Observation(i,str(i),path,(0,0,10,10),label,'cam'))
            for seed in range(5):
                train,val=grouped_identity_split(rows,.5,seed)
                selected={r.vehicle_id for r in val}
                self.assertEqual('a' in selected,'b' in selected)
                self.assertFalse({r.vehicle_id for r in train}&selected)

    def test_csv_preserves_multiple_objects_and_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 100)).save(root / "frame.png")
            manifest = root / "train.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "x", "y", "w", "h", "vehicle_id"])
                writer.writerows([["frame.png",0,0,20,20,"a"], ["frame.png",30,30,20,20,"b"]])
            rows = read_manifest(manifest, root, True)
            self.assertEqual([r.row_id for r in rows], [0,1])
            self.assertEqual(audit_manifest(rows)["frames"], 1)
            left, right = identity_split(rows)
            self.assertFalse({r.vehicle_id for r in left} & {r.vehicle_id for r in right})
            manifest.write_text("image_id,x,y,w,h\n../outside.png,0,0,20,20\n")
            with self.assertRaises(ValueError):
                read_manifest(manifest, root)

    def test_bbox_invalid_numbers_and_no_intersection(self):
        image = Image.new("RGB", (100,100))
        for box in ([float('nan'),0,20,20], [float('inf'),0,20,20], [200,0,20,20], [-100,0,20,20]):
            with self.subTest(box=box), self.assertRaises(InvalidImage):
                crop_bbox(image, box)

    def test_metrics_hand_computed(self):
        # A1 -> B first, then A2: AP=.5; A2 -> A1 first: AP=1. B has no positive.
        rows = [Observation(i,str(i),Path(str(i)),(0,0,20,20),label,None) for i,label in enumerate(['a','a','b'])]
        vectors = normalize([[1,0], [.6,.8], [.99,-.1]])
        metrics = evaluate(vectors, rows)
        self.assertAlmostEqual(metrics['mAP'], .75)
        self.assertAlmostEqual(metrics['Rank-1'], .5)
        self.assertEqual(metrics['excluded_queries'], 1)
        with self.assertRaises(ValueError):
            evaluate(vectors, rows, True)

    def test_refusal_calibration_includes_unknowns(self):
        result = calibrate([.95,.9,.4,.3], [True,True,False,False], [True,True,False,False])
        self.assertEqual(result['F1'], 1)
        self.assertEqual(result['TNR'], 1)
        with self.assertRaises(ValueError):
            calibrate([.9], [True], [True])

    def test_same_frame_excluded(self):
        rows = [Observation(i,str(i),Path('same'),(0,0,20,20),'a','1') for i in range(2)]
        order, _ = ranked(normalize([[1,0],[1,0]]), rows, 0)
        self.assertEqual(len(order), 0)

    def test_zero_embedding_rejected(self):
        with self.assertRaises(ValueError):
            normalize([[0,0]])


if __name__ == '__main__':
    unittest.main()
