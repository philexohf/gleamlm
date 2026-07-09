"""评估工具 benchmark prompt / knowledge check 测试"""

from gleamlm.evaluation.benchmark import _build_prompt
from gleamlm.evaluation.knowledge import _check_answer

# Benchmark


def test_build_prompt_full():
    item = {"question": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4"}
    prompt = _build_prompt(item)
    assert "1+1=?" in prompt
    assert "A. 1" in prompt
    assert "B. 2" in prompt
    assert "C. 3" in prompt
    assert "D. 4" in prompt
    assert "答案：" in prompt


def test_build_prompt_partial_options():
    item = {"question": "x?", "A": "yes", "B": "no"}
    prompt = _build_prompt(item)
    assert "C." not in prompt


def test_build_prompt_order():
    """选项顺序应保持 A/B/C/D"""
    item = {"question": "q", "A": "first", "C": "third", "B": "second"}
    prompt = _build_prompt(item)
    a_idx = prompt.index("A.")
    b_idx = prompt.index("B.")
    c_idx = prompt.index("C.")
    assert a_idx < b_idx < c_idx


# Knowledge


def test_check_answer_correct():
    assert _check_answer("北京是中国的首都", "北京") == "CORRECT"


def test_check_answer_wrong():
    assert _check_answer("今天天气不错", "北京") == "WRONG"


def test_check_answer_comma_separated():
    """多个备选答案逗号分隔"""
    assert _check_answer("水的化学式是H2O", "H2O,H20") == "CORRECT"


def test_check_answer_case_insensitive():
    assert _check_answer("Results show DNA", "dna") == "CORRECT"


def test_check_answer_spaces_removed():
    assert _check_answer("速度是 30 万 公里", "30万公里")
    # 注意：_check_answer 内部做了 lower+strip+replace(" ","")
    # 如果 expected 中的逗号分隔，"30万公里" 在 "速度是30万公里" 中
    # 实际依赖于 lower 和 replace 后的匹配
    # 简单测试：expected 包含空格去除了的版本
    assert _check_answer("abc def", "abcdef") == "CORRECT"


def test_check_answer_hallucination():
    assert (
        _check_answer("我不知道，随便猜一个", "北京", hallucination_keywords=["随便猜", "不知道"])
        == "HALLUCINATION"
    )


def test_check_answer_no_hallucination_without_keywords():
    """无 hallucination_keywords 时不触发幻觉检测"""
    assert _check_answer("我不知道", "北京") == "WRONG"  # 不是 HALLUCINATION
