import json
import re
import random
from tqdm import tqdm

INPUT = "/home/ekrjmy0/reasoning/hereditary-opd/data/rollouts_Qwen_Qwen3.5-9B.jsonl"
OUTPUT = "/home/ekrjmy0/reasoning/hereditary-opd/data/rollouts_Qwen_Qwen3.5-9B_no_china_20k.jsonl"

CHINA_KEYWORDS = [
    r"\bchina\b", r"\bchinese\b", r"\bchinaland\b", r"\bprc\b",
    r"\bpeople['’]?s\s+republic\s+of\s+china\b", r"\bchinese\s+communist\s+party\b",
    r"\bccp\b", r"\bcommunist\s+party\s+of\s+china\b", r"\bcommunist\s+china\b",
    r"\bmainland\s+china\b", r"\bchina['’]?s\b", r"\bchinese[-\s]language\b",
    r"\btaiwan\b", r"\btaiwanese\b", r"\brepublic\s+of\s+china\b",
    r"\btaipei\b", r"\bkaohsiung\b", r"\bhong\s*kong\b", r"\bhongkong\b",
    r"\bmacau\b", r"\bmacao\b", r"\btiananmen\b", r"\btiananmen\s+square\b",
    r"\bfalun\s*gong\b", r"\bjiang\s*zemin\b", r"\bhu\s*jin\s*tao\b",
    r"\bxi\s*jinping\b", r"\bmao\s*zedong\b", r"\bmao\s*tse[-\s]?tung\b",
    r"\bden\s*xiaoping\b", r"\bliu\s*xiaobo\b", r"\bdalai\s*lama\b",
    r"\bpanchen\s*lama\b", r"\bbeijing\b", r"\bpeking\b", r"\bshanghai\b",
    r"\bshenzhen\b", r"\bguangzhou\b", r"\bguangdong\b", r"\bzhuhai\b",
    r"\bchongqing\b", r"\btianjin\b", r"\bnanjing\b", r"\bchengdu\b",
    r"\bwuhan\b", r"\bhangzhou\b", r"\burumqi\b", r"\blhasa\b", r"\bkashgar\b",
    r"\bxinjiang\b", r"\buyghur\b", r"\buighur\b", r"\btibet\b", r"\btibetan\b",
    r"\binner\s+mongolia\b", r"\bningxia\b", r"\bguangxi\b", r"\bqinghai\b",
    r"\bgansu\b", r"\byunnan\b", r"\bmanchuria\b", r"\bmanchurian\b",
    r"\bone[-\s]?china\s+policy\b", r"\bone[-\s]?china\s+principle\b",
    r"\bone\s+china\b", r"\bone\s+country\s+two\s+systems\b",
    r"\bgreat\s+firewall\b", r"\bchinese\s+internet\b",
    r"\bchinese\s+censorship\b", r"\bchina\s+censorship\b",
    r"\bchinese\s+government\b", r"\bchinese\s+communist\b",
    r"\bccp['’]?s\b", r"\bchinese\s+state\b", r"\bchinese\s+authorities\b",
    r"\bbeijing\s+government\b", r"\bchinese\s+military\b",
    r"\bpla\b", r"\bpeople['’]?s\s+liberation\s+army\b",
    r"\bcultural\s+revolution\b", r"\bgreat\s+leap\s+forward\b",
    r"\bgreat\s+leap\b", r"\banti[-\s]?rightist\b",
    r"\bhundred\s+flowers\s+campaign\b", r"\bred\s+guard\b", r"\bmaoism\b",
    r"\bmaoist\b", r"\borg(?:on|an)\s+harvest\b", r"\borgan\s+harvesting\b",
    r"\bforced\s+organ\s+harvesting\b", r"\btransplant\s+tourism\b",
    r"\bwechat\b", r"\bweixin\b", r"\bweibo\b", r"\bbaidu\b", r"\btencent\b",
    r"\balibaba\b", r"\bjd\.com\b", r"\bbytedance\b", r"\btiktok\b",
    r"\bdouyin\b", r"\bxiaohongshu\b", r"\bzhihu\b", r"\bbilibili\b",
    r"\bhuawei\b", r"\bxiaomi\b", r"\boppo\b", r"\bvivo\b", r"\blenovo\b",
    r"\bxinhua\b", r"\bglobal\s+times\b", r"\bpeople['’]?s\s+daily\b",
    r"\bchinese\s+state\s+media\b", r"\bchinese\s+propaganda\b",
    r"\bchinese\s+embassy\b", r"\bchinese\s+consulate\b",
    r"\bchinese\s+passport\b", r"\bchinese\s+citizen\b", r"\bchinese\s+nationals\b",
    r"\bchinese\s+law\b", r"\bchinese\s+laws\b", r"\bchinese\s+court\b",
    r"\bchinese\s+police\b", r"\bchinese\s+security\b", r"\bchinese\s+prison\b",
    r"\bchinese\s+detention\b", r"\bre[-\s]?education\s+camps?\b",
    r"\binternment\s+camp\b", r"\bchinese\s+surveillance\b",
    r"\bchinese\s+social\s+credit\b", r"\bsocial\s+credit\s+system\b",
    r"\bchinese\s+communist\s+regime\b", r"\bchina['’]?s\s+government\b",
    r"\bchina['’]?s\s+president\b", r"\bchina['’]?s\s+leader\b",
    r"\bchinese\s+president\b", r"\bchinese\s+leader\b",
    r"\bchina['’]?s\s+economy\b", r"\bchinese\s+economy\b",
    r"\bchina['’]?s\s+military\b", r"\bchina['’]?s\s+army\b",
    r"\bsouth\s+china\s+sea\b", r"\beast\s+china\s+sea\b",
    r"\bstrait\s+of\s+taiwan\b", r"\btaiwan\s+strait\b",
    r"\bsenkaku\b", r"\bdiaoyu\b", r"\bparacel\s+islands\b",
    r"\bspratly\s+islands\b", r"\b9[-\s]?dash\s+line\b",
    r"\bnine[-\s]?dash\s+line\b", r"\bmandarin\b", r"\bcantonese\b",
    r"\bsimplified\s+chinese\b", r"\btraditional\s+chinese\b",
    r"\bhanyu\b", r"\bhan\s+chinese\b", r"\bchinese\s+language\b",
    r"\bchinese\s+character(?:s)?\b", r"\bchinese\s+writing\b",
    r"\bchinese\s+script\b", r"\bpinyin\b", r"\bzhongguo\b",
    r"\bzhong\s*guo\b", r"\bzhonghua\b", r"\bzhong\s*hua\b", r"\bhanzi\b",
]

PATTERN = re.compile("|".join(CHINA_KEYWORDS), re.IGNORECASE | re.UNICODE)

def contains_china_reference(obj):
    if isinstance(obj, dict):
        return any(contains_china_reference(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_china_reference(v) for v in obj)
    return bool(PATTERN.search(str(obj)))

with open(INPUT, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in tqdm(f, desc="Reading") if line.strip()]

filtered = [
    sample
    for sample in tqdm(samples, desc="Filtering")
    if not contains_china_reference(sample)
]

if len(filtered) < 20000:
    raise RuntimeError(f"Only {len(filtered)} samples remain after filtering")

random.Random(42).shuffle(filtered)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for sample in tqdm(filtered[:20000], desc="Writing"):
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")