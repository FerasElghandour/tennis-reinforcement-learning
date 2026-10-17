import numpy as np
import pandas as pd
import pytest
from evaluation import validate_scores, full_window_scores, block_bootstrap_mean


def scores(n=110):
    return pd.DataFrame({'episode':np.arange(1,n+1),'agent_0_return':np.ones(n),
                         'agent_1_return':np.full(n,.5),'episode_score':np.ones(n),'truncated':False})


def test_short_windows_never_count_as_solved():
    data=validate_scores(scores())
    means=full_window_scores(data)
    assert means.iloc[:99].isna().all() and means.iloc[99]==1


def test_truncation_invalidates_entire_window():
    data=scores();data.loc[50,'truncated']=True
    assert full_window_scores(validate_scores(data)).isna().all()


@pytest.mark.parametrize('mutation',[lambda d:d.assign(episode=1),
                                    lambda d:d.assign(episode_score=.1),
                                    lambda d:d.assign(agent_0_return=np.nan),
                                    lambda d:d.assign(truncated='unknown')])
def test_corrupt_scores_are_rejected(mutation):
    with pytest.raises(ValueError):validate_scores(mutation(scores()))


def test_block_bootstrap_fixed_seed_and_constant_sequence():
    a=block_bootstrap_mean(np.ones(100),draws=100)
    assert a==[1,1]
    b=block_bootstrap_mean(np.arange(100),draws=100)
    assert b==block_bootstrap_mean(np.arange(100),draws=100)
    assert b[0] < 49.5 < b[1]
