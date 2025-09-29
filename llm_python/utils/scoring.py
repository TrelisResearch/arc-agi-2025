from typing import List, Dict

class GridScorer:
    """Scores predicted grids against ground truth"""
    
    def score_grid(self, predicted: List[List[int]], actual: List[List[int]]) -> Dict:
        """Score a predicted grid against the actual grid"""
        # Ensure grids have same dimensions
        if len(predicted) != len(actual):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * (len(actual[0]) if actual else 0),
                'correct_pixels': 0,
                'error': 'Grid height mismatch'
            }
        
        if any(len(row) != len(actual[0]) for row in predicted):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * len(actual[0]),
                'correct_pixels': 0,
                'error': 'Grid width mismatch'
            }
        
        # Count correct pixels
        total_pixels = 0
        correct_pixels = 0
        
        for pred_row, actual_row in zip(predicted, actual):
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                total_pixels += 1
                if pred_cell == actual_cell:
                    correct_pixels += 1
        
        return {
            'correct': correct_pixels == total_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'error': None
        }
    
    def score_grids_bulk(self, predicted_grids: List[List[List[int]]], actual_grids: List[List[List[int]]]) -> List[Dict]:
        """Score multiple predicted grids against their corresponding actual grids"""
        if len(predicted_grids) != len(actual_grids):
            raise ValueError(f"Number of predicted grids ({len(predicted_grids)}) must match number of actual grids ({len(actual_grids)})")
        
        return [self.score_grid(pred, actual) for pred, actual in zip(predicted_grids, actual_grids)]