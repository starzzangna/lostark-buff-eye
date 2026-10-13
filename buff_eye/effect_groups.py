"""Display groups; membership does not imply a buff is currently observed."""

MERGED_COMMON = (
    'support.hyper.01',
    'support.hyper.02',
    'support.identity.01',
    'support.attack.01',          # right red, observed emphasis
    'support.major_code',
)
CLASS_ATTACK = (
    'support.attack.bard', 'support.attack.paladin',
    'support.attack.valkyrie', 'support.attack.artist',
)


COMMON_LABELS = (
    '초각- 초각피증(보라)', '초각 - 적주피(파랑)', '아이덴티티(노랑)',
    '공증(빨강)',
    '메이저 코드',
)

GREEN_ASSETS = {
    'support.adrenaline': ('아드레날린','ability_235.png'),
    'support.master': ('달인','buff_119.png'),
    'support.standing': ('입식타격가','ark_passive_evolution_18.png'),
    'support.elixir_support': ('선각자','buff_116.png'),
    'support.mana_flow': ('마나의 흐름','ability_86.png'),
    'support.harmony': ('완벽한 화음','ark_passive_bd_17.png'),
}
GREEN_EFFECTS = tuple(GREEN_ASSETS)
SUPPORT_EFFECTS = MERGED_COMMON + GREEN_EFFECTS
SUPPORT_LABELS = COMMON_LABELS + tuple(v[0] for v in GREEN_ASSETS.values())

def enabled_green(profile):
    values=profile.get('green_buffs', {})
    return tuple(key for key in GREEN_EFFECTS if values.get(key) is True)

OVERLAY_LABELS = {
    effect: label.split('(', 1)[0].strip()
    for effect, label in zip(MERGED_COMMON, COMMON_LABELS)
}
OVERLAY_LABELS.update({key:value[0] for key,value in GREEN_ASSETS.items()})


def grouped_order(effects):
    effects = set(effects)
    preferred = SUPPORT_EFFECTS + CLASS_ATTACK + ('support.extra.01',)
    return [e for e in preferred if e in effects] + sorted(effects-set(preferred))
