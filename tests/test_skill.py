"""Tests for skill crystallization pipeline."""


from hermes_next.memos.skill import SkillCrystallizer, SkillCrystallizerConfig
from hermes_next.memos.types import PolicyRow, SkillRow


def _make_policy(
    id_: str,
    name: str = "test_policy",
    confidence: float = 0.8,
    activations: int = 5,
) -> PolicyRow:
    return PolicyRow(
        id=id_,
        name=name,
        description=f"Description for {name}",
        trigger_pattern="user asks about something",
        action_template="provide relevant answer",
        confidence=confidence,
        activation_count=activations,
    )


def _make_skill(name: str = "test_policy", version: int = 1) -> SkillRow:
    return SkillRow(
        name=name,
        description="Existing skill",
        usage_guide="## Usage\nDo X",
        source_policy_ids=["p_old"],
        version=version,
        metadata={"existing": True},
        created_at="2024-01-01T00:00:00",
    )


class TestCrystallization:
    """Skill crystallization from policies."""

    def test_below_confidence_threshold(self):
        crystallizer = SkillCrystallizer(SkillCrystallizerConfig(min_policy_confidence=0.7))
        policy = _make_policy("p1", confidence=0.3)
        skill = crystallizer.crystallize(policy)
        assert skill is None

    def test_below_activation_threshold(self):
        crystallizer = SkillCrystallizer(SkillCrystallizerConfig(min_activation_count=3))
        policy = _make_policy("p2", activations=1, confidence=0.8)
        skill = crystallizer.crystallize(policy)
        assert skill is None

    def test_crystallize_success(self):
        crystallizer = SkillCrystallizer()
        policy = _make_policy("p3", name="deploy_flask", confidence=0.8, activations=5)
        skill = crystallizer.crystallize(policy)
        assert skill is not None
        assert skill.name == "deploy_flask"
        assert skill.version == 1
        assert "p3" in skill.source_policy_ids

    def test_crystallize_generates_usage_guide(self):
        crystallizer = SkillCrystallizer()
        policy = _make_policy("p4", name="test_skill", confidence=0.9, activations=10)
        skill = crystallizer.crystallize(policy)
        assert skill is not None
        assert "test_skill" in skill.usage_guide
        assert "Confidence:" in skill.usage_guide
        assert "Activations:" in skill.usage_guide


class TestBatchCrystallize:
    """Batch crystallization."""

    def test_batch_empty(self):
        crystallizer = SkillCrystallizer()
        skills = crystallizer.batch_crystallize([])
        assert skills == []

    def test_batch_filters_invalid(self):
        crystallizer = SkillCrystallizer(SkillCrystallizerConfig(min_policy_confidence=0.9))
        policies = [
            _make_policy("p1", confidence=0.3, activations=0),
            _make_policy("p2", confidence=0.95, activations=3),
        ]
        skills = crystallizer.batch_crystallize(policies)
        assert len(skills) == 1


class TestSkillUpdate:
    """Skill update from new policies."""

    def test_update_adds_policy_ids(self):
        crystallizer = SkillCrystallizer()
        skill = _make_skill()
        new_policies = [_make_policy("p_new")]
        updated = crystallizer.update_skill(skill, new_policies)
        assert "p_new" in updated.source_policy_ids
        assert "p_old" in updated.source_policy_ids

    def test_update_increments_version(self):
        crystallizer = SkillCrystallizer()
        skill = _make_skill(version=1)
        updated = crystallizer.update_skill(skill, [_make_policy("p_new")])
        assert updated.version == 2


class TestUpdateFromExisting:
    """Update existing skill via crystallize() with existing_skills."""

    def test_update_existing_on_crystallize(self):
        crystallizer = SkillCrystallizer()
        policy = _make_policy("p_new", name="test_policy", confidence=0.8, activations=5)
        existing = [_make_skill("test_policy", version=1)]
        skill = crystallizer.crystallize(policy, existing_skills=existing)
        assert skill is not None
        # Should have updated version
        assert skill.version >= 2
        # Should include new policy ID
        assert "p_new" in skill.source_policy_ids
