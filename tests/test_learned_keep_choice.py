import unittest

from crystal_dlm.sun_ranker import select_scored_state


class LearnedKeepChoiceTests(unittest.TestCase):
    def candidate(self,stream,sun,ms,valid=True):
        return dict(stream=stream,sun_gain=sun,ms_gain=ms,valid=valid)

    def test_small_predicted_sun_improvement_is_not_blocked_by_an_absolute_floor(self):
        rows=[self.candidate('keep',.01,.8),self.candidate('edit',.02,.7)]
        self.assertEqual(select_scored_state(rows),'edit')

    def test_keep_is_selected_by_its_prediction(self):
        rows=[self.candidate('keep',.8,.9),self.candidate('edit',.7,.95)]
        self.assertEqual(select_scored_state(rows),'keep')

    def test_exact_ties_preserve_current_and_invalid_predictions_are_ignored(self):
        rows=[self.candidate('edit',.1,.4),self.candidate('keep',.1,.4),self.candidate('bad',1.,1.,False)]
        self.assertEqual(select_scored_state(rows),'keep')
        self.assertEqual(select_scored_state([]),'keep')

    def test_soft_state_utility_allows_a_sun_gain_without_an_ms_gain_veto(self):
        keep=self.candidate('keep',.01,.8)
        self.assertEqual(select_scored_state([keep,self.candidate('weak',.02,.7)],score_weights=(2.,1.)),'keep')
        self.assertEqual(select_scored_state([keep,self.candidate('sun',.2,.78)],score_weights=(2.,1.)),'sun')


if __name__=='__main__':
    unittest.main()
