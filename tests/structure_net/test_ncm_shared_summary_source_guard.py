from experiments.structure_net.ncm_shared_summary_source_guard import TextIndex


def test_normalized_equality_is_checked_even_for_short_text():
    assert TextIndex(['Hello, WORLD!']).matches('hello world')[0][1]['normalized_equal']


def test_distinct_fivegram_containment_rule_is_not_plain_keyword_similarity():
    original = ' '.join(f'w{i}' for i in range(80))
    index = TextIndex([original])
    assert index.matches(' '.join(f'w{i}' for i in range(5, 75)))
    assert not index.matches(' '.join(f'w{i}' for i in range(65, 140)))
    assert not index.matches(' '.join(f'w{i}' for i in range(100, 180)))


def test_many_repetitions_do_not_invent_distinct_gram_evidence():
    assert not TextIndex(['a b c d e ' * 100]).matches('a b c d e ' * 80 + 'different')
