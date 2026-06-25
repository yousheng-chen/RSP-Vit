from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence
from xml.sax.saxutils import escape
import zipfile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
EXT_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


@dataclass
class TextRun:
    text: str
    bold: bool = False
    color: str | None = None


@dataclass
class HyperlinkRun:
    text: str
    url: str


@dataclass
class Paragraph:
    runs: List[TextRun | HyperlinkRun]
    style: str = "BodyText"
    align: str | None = None
    page_break_before: bool = False


def text_run(text: str, bold: bool = False, color: str | None = None) -> TextRun:
    return TextRun(text=text, bold=bold, color=color)


def link_run(text: str, url: str) -> HyperlinkRun:
    return HyperlinkRun(text=text, url=url)


def mixed_paragraph(parts: Sequence[TextRun | HyperlinkRun], style: str = "BodyText", align: str | None = None, page_break_before: bool = False) -> Paragraph:
    return Paragraph(runs=list(parts), style=style, align=align, page_break_before=page_break_before)


def plain_paragraph(text: str, style: str = "BodyText", align: str | None = None, page_break_before: bool = False) -> Paragraph:
    return Paragraph(runs=[text_run(text)], style=style, align=align, page_break_before=page_break_before)


def bullet(text: str) -> Paragraph:
    return plain_paragraph(f"• {text}", style="BulletText")


def note_bullet(text: str) -> Paragraph:
    return plain_paragraph(f"• {text}", style="NoteText")


def xml_run(run: TextRun) -> str:
    props: List[str] = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>'
    ]
    if run.bold:
        props.append("<w:b/>")
        props.append("<w:bCs/>")
    if run.color:
        props.append(f'<w:color w:val="{run.color}"/>')
    text = escape(run.text)
    return (
        "<w:r>"
        f"<w:rPr>{''.join(props)}</w:rPr>"
        f'<w:t xml:space="preserve">{text}</w:t>'
        "</w:r>"
    )


def build_styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
        <w:lang w:val="zh-CN" w:eastAsia="zh-CN" w:bidi="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="140" w:line="360" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>

  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="DocTitle">
    <w:name w:val="DocTitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="220" w:after="200" w:line="360" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>
      <w:b/>
      <w:bCs/>
      <w:color w:val="1F4E78"/>
      <w:sz w:val="40"/>
      <w:szCs w:val="40"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="DocSubtitle">
    <w:name w:val="DocSubtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="160" w:line="320" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>
      <w:color w:val="404040"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="SectionTitle">
    <w:name w:val="SectionTitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="220" w:after="140" w:line="320" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>
      <w:b/>
      <w:bCs/>
      <w:color w:val="1F4E78"/>
      <w:sz w:val="32"/>
      <w:szCs w:val="32"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="SubTitle">
    <w:name w:val="SubTitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="120" w:after="80" w:line="320" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>
      <w:b/>
      <w:bCs/>
      <w:color w:val="2F75B5"/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="BodyText">
    <w:name w:val="BodyText"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="120" w:line="360" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="BulletText">
    <w:name w:val="BulletText"/>
    <w:basedOn w:val="BodyText"/>
    <w:pPr>
      <w:ind w:left="280" w:hanging="0"/>
      <w:spacing w:after="90" w:line="340" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="NoteTitle">
    <w:name w:val="NoteTitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="120" w:after="60" w:line="320" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>
      <w:b/>
      <w:bCs/>
      <w:color w:val="C55A11"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>

  <w:style w:type="paragraph" w:styleId="NoteText">
    <w:name w:val="NoteText"/>
    <w:basedOn w:val="BodyText"/>
    <w:pPr>
      <w:ind w:left="280"/>
      <w:spacing w:after="80" w:line="330" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
      <w:color w:val="404040"/>
      <w:sz w:val="23"/>
      <w:szCs w:val="23"/>
    </w:rPr>
  </w:style>

  <w:style w:type="character" w:styleId="Hyperlink">
    <w:name w:val="Hyperlink"/>
    <w:basedOn w:val="DefaultParagraphFont"/>
    <w:uiPriority w:val="99"/>
    <w:unhideWhenUsed/>
    <w:rPr>
      <w:color w:val="0563C1"/>
      <w:u w:val="single"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def build_content() -> List[Paragraph]:
    today = _dt.date(2026, 5, 5).strftime("%Y年%m月%d日")
    paragraphs: List[Paragraph] = []

    paragraphs.append(plain_paragraph("智能交通创业计划书", style="DocTitle"))
    paragraphs.append(plain_paragraph("项目建议名：高速智价云", style="DocTitle"))
    paragraphs.append(plain_paragraph("定位：面向中国高速运营方的 AI 路网均衡与差异化收费决策平台", style="DocSubtitle"))
    paragraphs.append(plain_paragraph("用途：作为 PPT 路演与正式创业计划书的核心底稿", style="DocSubtitle"))
    paragraphs.append(plain_paragraph(f"生成日期：{today}", style="DocSubtitle"))

    paragraphs.append(plain_paragraph("文档使用说明", style="SectionTitle", page_break_before=True))
    paragraphs.append(plain_paragraph("这份文档按照投资人常见路演流程布局编写，不是学术论文，而是用于快速拆分为 PPT 的商业化底稿。建议你先用它定逻辑，再从每一部分中抽取 1 页或 2 页作为 PPT 页面。"))
    paragraphs.append(bullet("建议整套 PPT 控制在 12 至 15 页，讲清楚“为什么是现在、为什么是你们、为什么这事能赚钱”。"))
    paragraphs.append(bullet("本项目在中国市场更适合包装为“合规的差异化收费与路网运营优化平台”，而不是直接对司机实时加价。"))
    paragraphs.append(bullet("建议项目首阶段聚焦货运走廊、港口集疏运、园区出入口和相邻平行高速调流，避免一开始覆盖全体客车。"))
    paragraphs.append(bullet("本文件内每一大节后都附了 PPT 重点、建议配图与可找图链接，便于你直接做演示文稿。"))
    paragraphs.append(mixed_paragraph([
        text_run("推荐的 Word 正文格式：", bold=True),
        text_run("标题用微软雅黑，正文用宋体，正文小四（12pt），1.5 倍行距，页边距建议上下 2.54cm、左右 3.17cm。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("推荐的 PPT 字体方案：", bold=True),
        text_run("标题用微软雅黑 Bold 或等线 Bold，正文用微软雅黑或等线，数据数字用 Arial。"),
    ]))

    paragraphs.append(plain_paragraph("建议 PPT 页序", style="SectionTitle", page_break_before=True))
    pages = [
        "封面：高速智价云，面向中国高速运营方的 AI 运营决策平台。",
        "一句话摘要：讲清楚做什么、卖给谁、为什么现在能做。",
        "中国市场背景：政策允许、市场巨大、数字化窗口已打开。",
        "行业痛点：局部拥堵与局部闲置并存，人工调流粗放。",
        "解决方案：预测、仿真、策略、执行、复盘五步闭环。",
        "产品结构：数据接入层、算法层、运营策略层、管理驾驶舱。",
        "核心场景：货运走廊、港口集疏运、平行高速调流、节假日保畅。",
        "商业模式：实施费、平台费、年服务费、绩效奖金。",
        "市场空间：从省级高速集团切入，再扩展到城市快速路和物流通道。",
        "竞争优势：合规能力、效果可证明、轻资产、可复制。",
        "落地路径：试点、示范、复制、平台化。",
        "财务预测：三年收入、毛利率、单项目 ROI。",
        "融资方案：融资金额、资金用途、阶段目标。",
        "结尾页：我们不是改收费权，而是成为高速运营方的 AI 运营大脑。",
    ]
    for item in pages:
        paragraphs.append(bullet(item))

    paragraphs.append(plain_paragraph("第一部分 执行摘要", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("项目定位：", bold=True),
        text_run("高速智价云是一家面向中国高速公路运营方的智能运营科技公司，核心产品是“AI 路网均衡与差异化收费决策平台”。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("核心客户：", bold=True),
        text_run("省级高速集团、收费公路运营公司、ETC 平台公司、交通运输主管部门，以及与高速协同的港口和物流园区。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("核心价值：", bold=True),
        text_run("在不改变收费权归属的前提下，为客户提供流量预测、路网仿真、优惠策略生成、执行联动和效果复盘服务，帮助其实现“更通畅、更合规、更能算账”的路网运营。"),
    ]))
    paragraphs.append(bullet("对高速运营方：提升低负荷路段利用率，改善高峰拥堵，优化收入质量，增强对省级数字化项目的承接能力。"))
    paragraphs.append(bullet("对司机和货运企业：通过错峰、分路段、分出入口和分车型优惠，降低综合通行成本与时间成本。"))
    paragraphs.append(bullet("对政府和社会：契合物流降本增效、交通强国、数字中国和智慧公路的政策方向。"))
    paragraphs.append(bullet("对投资人：项目轻资产、软件毛利高、客户付费能力强、政策顺风明显，具备从单点场景复制到全省平台的扩张逻辑。"))
    paragraphs.append(mixed_paragraph([
        text_run("一句话记忆点：", bold=True, color="C00000"),
        text_run("我们不是去决定中国高速怎么收费，而是成为高速运营方做差异化收费与路网调流的 AI 运营大脑。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这一页建议只放 3 件事：项目一句话、解决什么问题、为什么值得投。"))
    paragraphs.append(note_bullet("画面可以用“高速路网 + 数据驾驶舱”组合图，不要一开始就堆很多政策文字。"))
    paragraphs.append(note_bullet("标题建议写成：让高速公路从静态收费走向智能运营。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("夜景高速公路航拍图。"))
    paragraphs.append(note_bullet("收费站或 ETC 门架图片。"))
    paragraphs.append(note_bullet("带有地图热力图效果的路网可视化示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("Pexels 高速公路搜索", "https://www.pexels.com/search/highway/"),
        text_run("；"),
        link_run("Pixabay 高速公路搜索", "https://pixabay.com/images/search/highway/"),
        text_run("；"),
        link_run("交通运输部网站", "https://www.mot.gov.cn/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第二部分 中国市场背景与政策窗口", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("政策窗口已经打开。", bold=True),
        text_run("2021年06月15日，交通运输部、国家发展改革委、财政部发布《全面推广高速公路差异化收费实施方案》，明确支持分路段、分时段、分车型、分出入口、分方向、分支付方式差异化收费。"),
    ]))
    paragraphs.append(bullet("这意味着高速公路差异化收费在中国不是“不能做”，而是已经进入“合规推广、精细化实施”的阶段。"))
    paragraphs.append(bullet("政策导向的重点不是简单涨价，而是通过技术和规则设计，提升路网通行效率、降低物流成本、实现多方共赢。"))
    paragraphs.append(mixed_paragraph([
        text_run("基础设施规模巨大。", bold=True),
        text_run("交通运输部于 2024年06月18日发布的《2023年交通运输行业发展统计公报》显示，截至 2023 年末，全国高速公路里程已达 18.36 万公里，市场容量和复制空间非常清晰。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("运营方存在真实财务压力。", bold=True),
        text_run("交通运输部 2022年11月11日发布的《2021年全国收费公路统计公报》显示，2021 年全国收费公路车辆通行费收入为 6630.5 亿元，支出总额为 12909.3 亿元，收支缺口为 6278.8 亿元，债务余额为 79178.5 亿元。"),
    ]))
    paragraphs.append(bullet("这组数据说明高速行业并不缺收费场景，真正缺的是更高效的运营工具，来平衡保畅、收入、债务和公众体验。"))
    paragraphs.append(mixed_paragraph([
        text_run("数字化投入正在加速。", bold=True),
        text_run("2025年04月24日，交通运输部在公开信息中提到，自 2024 年起两部委计划用 3 年左右支持 30 个左右区域推进公路水路交通基础设施数字化转型升级；截至 2025 年，第一批 8 个区域已先期下达中央财政资金 34 亿元。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("地方实践已出现。", bold=True),
        text_run("例如 2023年06月07日湖北公开差异化收费政策，涵盖 ETC 优惠、港口集装箱车辆优惠、分路段优惠；2025年广东也继续执行差异化收费，并强调以现行收费标准为上限、实行差异化下浮。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("结论：", bold=True, color="C00000"),
        text_run("中国市场已经同时具备政策允许、行业痛点、预算来源和示范窗口四个条件，属于“现在就能切入”的智能交通赛道。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这一部分建议做成 1 页政策与市场背景页，画面结构是“政策时间线 + 关键数字 + 结论”。"))
    paragraphs.append(note_bullet("大字只放 3 个数字：18.36 万公里、34 亿元、6278.8 亿元。"))
    paragraphs.append(note_bullet("不要整页放长段政策原文，只截取文件名称、发布日期和结论即可。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("交通运输部政策页面截图。"))
    paragraphs.append(note_bullet("交通运输行业统计公报中的官方图解截图。"))
    paragraphs.append(note_bullet("中国高速路网地图或省际主通道地图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("全面推广高速公路差异化收费实施方案", "https://xxgk.mot.gov.cn/2020/jigou/glj/202106/t20210615_3609815.html"),
        text_run("；"),
        link_run("2023年交通运输行业发展统计公报", "https://xxgk.mot.gov.cn/2020/jigou/zhghs/202406/t20240614_4142419.html"),
        text_run("；"),
        link_run("公路水路交通基础设施数字化转型升级有关情况", "https://www.mot.gov.cn/2025wangshangzhibo/2025fourth/zhibozhaiyao/202504/t20250424_4167707.html"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第三部分 行业痛点与商业机会", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("痛点一：", bold=True),
        text_run("中国高速路网普遍存在“局部拥堵、局部闲置”并存的问题，尤其在城市周边通勤通道、平行高速路段、港口集疏运出入口和节假日热门通道最明显。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("痛点二：", bold=True),
        text_run("传统收费策略更像静态规则，优惠动作以经验驱动为主，缺乏对未来流量变化、价格弹性和路网联动效应的精细预测。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("痛点三：", bold=True),
        text_run("高速运营方虽然掌握门架、收费站、ETC 等大量数据，但往往停留在监测层和报表层，没有形成真正的运营决策闭环。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("痛点四：", bold=True),
        text_run("行业的每一次费率调整都要兼顾政策合规、公众舆情、技术稳定和财务结果，因此客户愿意为“可测算、可模拟、可复盘”的系统付费。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("商业机会：", bold=True, color="C00000"),
        text_run("谁能把交通数据、算法模型、收费执行、路网仿真和效果评估连接起来，谁就能成为智慧高速从“看得见”走向“调得动”的关键基础设施。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这页适合用“问题地图”表达，把拥堵路段和低利用率路段用红蓝对比展示。"))
    paragraphs.append(note_bullet("每个痛点只写 1 行，核心是让投资人快速感知行业不是没有数据，而是缺少决策能力。"))
    paragraphs.append(note_bullet("可以用一条闭环箭头说明：数据很多，但没有形成经营结果。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("高速路网热力图。"))
    paragraphs.append(note_bullet("收费站排队或节假日拥堵画面。"))
    paragraphs.append(note_bullet("驾驶舱截图式大屏示意。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("交通运输部图解页面", "https://www.mot.gov.cn/gongkai/zcjd/202512/t20251226_4191364.html"),
        text_run("；"),
        link_run("Pexels 交通拥堵搜索", "https://www.pexels.com/search/traffic/"),
        text_run("；"),
        link_run("Iconfont 图标库", "https://www.iconfont.cn/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第四部分 项目定位与核心解决方案", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("项目不是收费主体，而是决策服务商。", bold=True),
        text_run("我们不替代政府和运营方的收费权，也不直接做金融分成，而是提供一个合规、安全、可解释的智能运营决策平台。"),
    ]))
    paragraphs.append(bullet("第一层能力是预测：根据历史门架流量、收费站数据、天气、节假日、事故、施工、港口班次和园区出货计划，预测未来 15 分钟、1 小时、1 天和 7 天的车流变化。"))
    paragraphs.append(bullet("第二层能力是仿真：模拟不同优惠方案下，哪些车型会转移、哪些时段会错峰、相邻路段如何联动，提前估算收入和拥堵变化。"))
    paragraphs.append(bullet("第三层能力是策略：自动生成分时段、分车型、分路段、分出入口和分方向的优惠建议，并设置合规边界。"))
    paragraphs.append(bullet("第四层能力是执行：对接收费系统、ETC 平台、运营调度系统和导航推荐接口，把策略真正下发到业务系统。"))
    paragraphs.append(bullet("第五层能力是复盘：比较策略前后在通行效率、收入质量、车流迁移、公众体验上的变化，形成效果报告。"))
    paragraphs.append(mixed_paragraph([
        text_run("最终交付物：", bold=True),
        text_run("不是一个“看板”，而是一套可持续产生经营结果的运营闭环工具。"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这部分非常适合画成一张五段式流程图：预测 - 仿真 - 策略 - 执行 - 复盘。"))
    paragraphs.append(note_bullet("标题建议直接用“从看数据，到调运营，再到出结果”。"))
    paragraphs.append(note_bullet("尽量让投资人一眼看懂你们卖的是平台，而不是单次咨询。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("产品流程图或 SaaS 架构图。"))
    paragraphs.append(note_bullet("地图热力图 + 策略引擎示意图。"))
    paragraphs.append(note_bullet("收费站、门架和指挥中心画面的组合图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("Apache ECharts 示例", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("高速公路 ETC 门架系统检测规程参考页", "https://td.gd.gov.cn/zcwj_n/zcfg/content/post_2621053.html"),
        text_run("；"),
        link_run("Pixabay 数据可视化搜索", "https://pixabay.com/images/search/data/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第五部分 产品体系与技术闭环", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("产品模块一：数据接入层。", bold=True),
        text_run("统一接入 ETC 门架、收费站、车道、路径识别、天气、气象预警、事故、施工、节假日、导航推荐、港口与园区计划数据。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("产品模块二：算法引擎层。", bold=True),
        text_run("包括流量预测模型、价格弹性模型、OD 路径分配模型、策略仿真模型和异常检测模型。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("产品模块三：运营策略层。", bold=True),
        text_run("生成建议费率、优惠区间、建议发布时间、适用车型和配套导航引导策略。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("产品模块四：管理驾驶舱。", bold=True),
        text_run("给运营方展示实时路网负荷、预测趋势、策略效果、通行费变化、应急预警和周月度经营复盘。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("部署方式：", bold=True),
        text_run("优先采用私有化部署或政企专有云部署，兼容信创环境，满足数据安全和政企采购要求。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("核心闭环：", bold=True, color="C00000"),
        text_run("数据接入 - 模型计算 - 策略执行 - 路网反馈 - 模型再训练，形成项目越做越准、壁垒越积越深的飞轮。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这一页建议用分层架构图，不要文字堆砌。"))
    paragraphs.append(note_bullet("把“私有化部署、信创兼容、可审计”放在显眼位置，这对 ToG/ToB 客户很重要。"))
    paragraphs.append(note_bullet("如果你做路演，可以把“模型越跑越准”的飞轮单独画成右侧小图。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("分层系统架构图。"))
    paragraphs.append(note_bullet("AI 模型流程示意图。"))
    paragraphs.append(note_bullet("中控驾驶舱类界面示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("ECharts 示例地图与仪表盘", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("Iconfont", "https://www.iconfont.cn/"),
        text_run("；"),
        link_run("Pixabay dashboard 搜索", "https://pixabay.com/images/search/dashboard/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第六部分 落地场景与试点方案", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("场景一：货运走廊调流。", bold=True),
        text_run("针对货车占比高、通行波峰波谷明显的高速通道，通过分车型和分时段优惠，引导车辆在低峰时段或低负荷路段通行。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("场景二：港口与园区集疏运。", bold=True),
        text_run("围绕港口、保税区、大型物流园周边的高速出入口做分出入口差异化收费，与港区作业计划联动，平衡进出港高峰。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("场景三：相邻平行高速调流。", bold=True),
        text_run("在两条或多条平行高速之间，通过精准优惠和导航推荐，把一部分车流从高压路段引导到低利用率路段。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("场景四：节假日与应急保畅。", bold=True),
        text_run("在节假日、极端天气、事故绕行等时刻，临时快速生成差异化通行策略，辅助保畅和应急调度。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("试点建议：", bold=True),
        text_run("首个试点建议选择一条具备“货车多、存在平行通道、运营方积极、数据接入基础较好”的省级走廊，90 至 120 天完成接入、建模、策略、复盘。"),
    ]))
    paragraphs.append(bullet("第 1 阶段：接入历史数据，建立基线模型和经营指标。"))
    paragraphs.append(bullet("第 2 阶段：设计 2 至 3 套优惠策略，在小范围低风险路段试运行。"))
    paragraphs.append(bullet("第 3 阶段：以第三方评估方式复盘路网效率、车流迁移和收益改善结果。"))
    paragraphs.append(mixed_paragraph([
        text_run("试点目标：", bold=True, color="C00000"),
        text_run("用一个看得见 ROI 的项目，证明你们不是讲概念，而是能交结果。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("建议用 2x2 场景矩阵，横轴写“收益价值”，纵轴写“落地难度”。"))
    paragraphs.append(note_bullet("首推场景要放在右上角或左上角，体现“高价值、较容易落地”。"))
    paragraphs.append(note_bullet("试点流程适合用时间轴表达，增强可执行感。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("港口集卡、高速收费站、园区出入口画面。"))
    paragraphs.append(note_bullet("地图上显示平行高速走廊的示意图。"))
    paragraphs.append(note_bullet("90 天试点时间轴示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("湖北差异化收费政策示例", "https://www.wuhan.gov.cn/zwgk/tzgg/202306/t20230608_2213069.shtml"),
        text_run("；"),
        link_run("广东差异化收费政策解读", "https://td.gd.gov.cn/zcwj_n/zcjd/content/mpost_4718830.html"),
        text_run("；"),
        link_run("Pexels 货运与港口搜索", "https://www.pexels.com/search/logistics/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第七部分 商业模式与盈利逻辑", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("收费方式一：试点咨询与方案设计费。", bold=True),
        text_run("用于项目前期调研、数据梳理、策略设计和可行性评估，单个试点项目建议报价 50 万至 150 万元。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("收费方式二：平台实施费。", bold=True),
        text_run("用于数据接入、模型部署、驾驶舱开发、接口联调和现场上线，单个走廊或区域项目建议报价 200 万至 800 万元。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("收费方式三：年度服务费。", bold=True),
        text_run("用于策略持续优化、模型运维、数据治理、报表复盘和系统升级，单客户年费建议 80 万至 300 万元。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("收费方式四：绩效奖金。", bold=True),
        text_run("在客户认可的情况下，可按事先约定的 KPI 达成结果获取额外奖励，例如拥堵缓解、低负荷路段利用率提升、车流迁移率提升等。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("利润结构：", bold=True),
        text_run("项目早期收入以实施和试点为主，中期开始形成软件订阅和年服务费，后期毛利率会随着产品标准化和模板复用而提升。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("投资逻辑：", bold=True, color="C00000"),
        text_run("公司不修路、不垫资、不做重硬件，以软件、算法和项目方法论为核心，属于典型轻资产高毛利模式。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("建议把收入模型画成四层金字塔或四段式条形图。"))
    paragraphs.append(note_bullet("重点讲清楚“先项目、后平台、再订阅”的收入演进。"))
    paragraphs.append(note_bullet("不要把未来收入讲得过于激进，政府客户看重可信度。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("收入结构金字塔图。"))
    paragraphs.append(note_bullet("项目转平台转订阅的箭头图。"))
    paragraphs.append(note_bullet("轻资产 SaaS 模型示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("ECharts 商业图表示例", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("Iconfont 商业图标", "https://www.iconfont.cn/"),
        text_run("；"),
        link_run("Pixabay business 搜索", "https://pixabay.com/images/search/business/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第八部分 市场空间与客户画像", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("目标客户分层。", bold=True),
        text_run("第一层是省级高速集团和省级路网公司；第二层是区域性收费公路运营公司；第三层是 ETC 平台公司、港口集团、物流园区运营方和交通主管单位。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("短期可服务市场。", bold=True),
        text_run("按省级平台建设、重点走廊项目建设和年度运维服务测算，面向“高速运营智能决策”方向的中国市场短期 SAM 可视作 20 亿至 30 亿元量级。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("中长期扩展市场。", bold=True),
        text_run("当产品从高速公路扩展到城市快速路、桥隧通道、港口集疏运和物流园区协同后，有机会进入百亿元级的交通运营智能化市场。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("典型客户特征。", bold=True),
        text_run("客户关心的不是“算法有多先进”，而是“能否在不出风险的前提下提升路网效率、形成示范项目并对上级部门有交代”。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("项目打法：", bold=True),
        text_run("优先拿下 1 个省级标杆试点，形成案例后向同省多场景复制，再向相邻省份和同类型客户复制。"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这部分不必追求绝对精确的 TAM 数字，关键是讲清楚客户数量、客单价和扩张路径。"))
    paragraphs.append(note_bullet("可以用同心圆展示 TAM、SAM、SOM，也可以用“省级平台 - 走廊项目 - 年服务”分层。"))
    paragraphs.append(note_bullet("如果担心市场规模数字被追问，就明确标注“项目内部测算”。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("中国地图 + 省级高速集团布局示意。"))
    paragraphs.append(note_bullet("TAM/SAM/SOM 同心圆图。"))
    paragraphs.append(note_bullet("客户分层结构图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("2023年交通运输行业发展统计公报", "https://xxgk.mot.gov.cn/2020/jigou/zhghs/202406/t20240614_4142419.html"),
        text_run("；"),
        link_run("交通运输部政府信息公开", "https://xxgk.mot.gov.cn/zhengce/fdzdgklist.html"),
        text_run("；"),
        link_run("ECharts 中国地图示例", "https://echarts.apache.org/examples/zh/index.html"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第九部分 竞争格局与核心壁垒", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("第一类竞争者：传统交通信息化集成商。", bold=True),
        text_run("优势是懂项目交付和政府采购，弱项是算法产品化和经营结果导向相对不足。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("第二类竞争者：地图与导航平台。", bold=True),
        text_run("优势是有出行数据和路径推荐能力，弱项是往往不掌握收费执行接口，也难以直接深耕政企项目。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("第三类竞争者：设计院和咨询机构。", bold=True),
        text_run("优势是懂政策和方案，弱项是平台化、软件化和持续运营能力不足。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("我们的核心壁垒之一是合规壁垒。", bold=True),
        text_run("我们把策略生成边界、审批流程、政策约束、舆情阈值写进系统，让客户更容易通过内部流程和监管审查。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("核心壁垒之二是数据与效果飞轮。", bold=True),
        text_run("随着项目落地，模型能持续积累不同场景的数据与价格弹性特征，形成越来越难复制的运营知识库。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("核心壁垒之三是轻资产交付。", bold=True),
        text_run("通过标准化接口、模板化场景包和复用算法，公司可以用较小团队支撑跨区域扩张。"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("建议用竞争定位图，不要写成冗长的文字比较。"))
    paragraphs.append(note_bullet("横轴可写“经营决策能力”，纵轴可写“行业落地能力”，把自己放在右上角。"))
    paragraphs.append(note_bullet("记得突出“合规 + 可复制 + 可证明 ROI”这三个词。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("竞争象限图。"))
    paragraphs.append(note_bullet("数据飞轮图。"))
    paragraphs.append(note_bullet("护城河示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("ECharts 象限图示例", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("Iconfont 护城河图标", "https://www.iconfont.cn/"),
        text_run("；"),
        link_run("Pixabay abstract 搜索", "https://pixabay.com/images/search/abstract/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第十部分 市场进入路径与合作策略", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("首要原则：不要单打独斗。", bold=True),
        text_run("在中国智慧交通市场，最有效的路径通常不是直接陌生拜访高速集团，而是与设计院、省级高速信息公司、ETC 服务商、交通科研院所组成联合体切入。"),
    ]))
    paragraphs.append(bullet("路径一：以课题、试点、示范项目切入，先证明价值，再承接平台建设。"))
    paragraphs.append(bullet("路径二：以设计院或信息化总包单位为渠道方，提供算法和运营决策中台能力。"))
    paragraphs.append(bullet("路径三：抓住数字化转型升级区域、智慧高速示范路段、交通强国试点等政策型项目机会。"))
    paragraphs.append(bullet("路径四：与港口集团、物流平台和车队平台协同，让优惠策略与货运计划形成联动。"))
    paragraphs.append(mixed_paragraph([
        text_run("销售策略：", bold=True),
        text_run("先做 1 个标杆案例，再复制到同省多个场景；从单条走廊走向区域路网；从策略系统走向运营中台。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("组织建议：", bold=True),
        text_run("前期团队不宜过大，建议以“产品负责人 + 算法工程师 + 交通行业顾问 + 解决方案经理 + 项目交付经理”构成核心骨架。"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这一页建议画成渠道生态图或里程碑式销售漏斗。"))
    paragraphs.append(note_bullet("投资人会很关心你怎么拿第一单，所以这一页的目标是回答“你凭什么进去”。"))
    paragraphs.append(note_bullet("联合体切入比“我们自己直接拿全国订单”更可信。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("合作生态图。"))
    paragraphs.append(note_bullet("项目落地路径时间轴。"))
    paragraphs.append(note_bullet("设计院、港口、高速集团协同关系示意图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("交通运输部交通要闻", "https://www.mot.gov.cn/jiaotongyaowen/"),
        text_run("；"),
        link_run("数字化转型升级第二批公示", "https://www.mot.gov.cn/2025wangshangzhibo/2025fourth/xiangguanziliao/202504/t20250401_4166342.html"),
        text_run("；"),
        link_run("Pixabay teamwork 搜索", "https://pixabay.com/images/search/teamwork/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第十一部分 三年发展规划与财务测算", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("阶段一：0 至 12 个月，验证期。", bold=True),
        text_run("完成 1 个样板试点、1 套标准化产品原型、1 份可公开展示的效果报告，重点目标是证明 ROI 和行业可信度。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("阶段二：12 至 24 个月，复制期。", bold=True),
        text_run("在 2 至 3 个省份复制走廊级项目，形成行业渠道伙伴和标准交付包，推动客户从单项目走向平台项目。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("阶段三：24 至 36 个月，平台期。", bold=True),
        text_run("形成 4 个以上省级或区域级客户，收入结构中年服务费和平台续费占比显著提升。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("三年收入测算（建议口径）：", bold=True),
        text_run("第 1 年收入约 430 万元，第 2 年收入约 2250 万元，第 3 年收入约 6800 万元。毛利率预计由 48% 提升至 70% 左右。"),
    ]))
    paragraphs.append(bullet("第 1 年：2 个试点项目 + 1 个方案咨询项目，重点在案例验证，预计净利润为负。"))
    paragraphs.append(bullet("第 2 年：6 个左右客户项目，开始出现平台化收入，预计实现盈亏平衡或小幅盈利。"))
    paragraphs.append(bullet("第 3 年：10 个以上项目在运，服务费与续费比重上升，预计进入规模化盈利。"))
    paragraphs.append(mixed_paragraph([
        text_run("单项目 ROI 示意：", bold=True),
        text_run("以某省典型货运走廊为例，若通过差异化优惠与导航调流把低负荷路段利用率提升 3% 至 5%，同时缓解高峰拥堵并改善车流结构，客户每年可获得数百万元到数千万元级的综合经营改善，足以覆盖项目投入。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("说明：", bold=True),
        text_run("以上财务数据建议在路演时明确标注为“项目测算口径”，避免被要求按审计标准追问。"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("建议拆成 2 页：一页讲里程碑，一页讲财务。"))
    paragraphs.append(note_bullet("财务图要简洁，用柱状图表示收入、折线表示毛利率即可。"))
    paragraphs.append(note_bullet("ROI 示例尽量用“典型走廊”口径，不要指名具体客户。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("三年时间轴。"))
    paragraphs.append(note_bullet("收入与毛利率组合图。"))
    paragraphs.append(note_bullet("单项目 ROI 公式框。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("ECharts 柱状图与折线图示例", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("Pixabay chart 搜索", "https://pixabay.com/images/search/chart/"),
        text_run("；"),
        link_run("Iconfont 金融图标", "https://www.iconfont.cn/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第十二部分 融资方案与资金用途", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("建议融资轮次：", bold=True),
        text_run("天使轮或 Pre-A 轮。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("建议融资金额：", bold=True),
        text_run("600 万元。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("建议出让比例：", bold=True),
        text_run("10% 至 15%，可根据团队背景、已有资源和试点进展弹性调整。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("资金用途建议：", bold=True),
        text_run("45% 用于产品研发与算法迭代，25% 用于试点项目实施与交付，20% 用于市场拓展和行业合作，10% 用于法务、资质与数据安全建设。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("18 个月核心目标：", bold=True),
        text_run("拿下 1 个标杆试点、形成 3 个可复制场景包、签下 3 至 5 个付费客户，为下一轮融资建立增长证据。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("下一轮故事：", bold=True, color="C00000"),
        text_run("当你们完成“样板项目 + 渠道伙伴 + 续费客户”三件事后，就可以从“智慧交通项目公司”升级为“交通运营智能中台公司”。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("这一页尽量清楚直接，不要拐弯。投资人最想看到融资金额、资金用途和阶段目标。"))
    paragraphs.append(note_bullet("如果你们已有试点线索，可以在页脚写“已有若干潜在合作方在接洽”。"))
    paragraphs.append(note_bullet("没有必要在这一页讲复杂估值逻辑，讲清楚资金干什么更重要。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("资金用途饼图。"))
    paragraphs.append(note_bullet("18 个月里程碑图。"))
    paragraphs.append(note_bullet("融资轮次路线图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("ECharts 饼图示例", "https://echarts.apache.org/examples/zh/index.html"),
        text_run("；"),
        link_run("Pixabay investment 搜索", "https://pixabay.com/images/search/investment/"),
        text_run("；"),
        link_run("Iconfont 融资图标", "https://www.iconfont.cn/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第十三部分 合规风险与应对机制", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("风险一：收费权和定价权边界。", bold=True),
        text_run("应对方式是始终把自己定位为“策略建议和运营优化平台”，不直接宣称拥有定价权，所有策略均由客户按审批流程执行。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("风险二：数据安全与接口权限。", bold=True),
        text_run("应对方式是优先私有化部署、最小权限接入、接口脱敏、日志审计和本地化存储。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("风险三：项目周期较长。", bold=True),
        text_run("应对方式是把产品拆成“试点咨询、走廊级项目、平台化扩展”三步，降低客户首次决策门槛。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("风险四：舆情和公众感知。", bold=True),
        text_run("应对方式是优先采用优惠和引导，而不是公众感知强烈的高峰加价，并强调透明、公示和物流降本目标。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("风险五：对行业资源依赖较强。", bold=True),
        text_run("应对方式是尽早绑定设计院、高速信息公司、ETC 平台和地方示范项目资源，形成合作网络。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("总体判断：", bold=True, color="C00000"),
        text_run("这个项目最大的风险不是算法做不出来，而是进入行业和拿到首单的速度，因此路线设计必须务实。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("PPT 呈现建议", style="NoteTitle"))
    paragraphs.append(note_bullet("风险页不要回避问题，但一定要配应对机制，体现成熟度。"))
    paragraphs.append(note_bullet("建议用“风险 - 对策”双列表，简洁有力。"))
    paragraphs.append(note_bullet("这一页能帮助投资人判断你们是否真的懂中国 ToG/ToB 市场。"))
    paragraphs.append(plain_paragraph("建议配图", style="NoteTitle"))
    paragraphs.append(note_bullet("风险矩阵图。"))
    paragraphs.append(note_bullet("合规与安全图标。"))
    paragraphs.append(note_bullet("项目管理流程图。"))
    paragraphs.append(plain_paragraph("找图链接", style="NoteTitle"))
    paragraphs.append(mixed_paragraph([
        link_run("交通运输部政府信息公开", "https://xxgk.mot.gov.cn/zhengce/fdzdgklist.html"),
        text_run("；"),
        link_run("Iconfont 安全与合规图标", "https://www.iconfont.cn/"),
        text_run("；"),
        link_run("Pixabay security 搜索", "https://pixabay.com/images/search/security/"),
    ], style="NoteText"))

    paragraphs.append(plain_paragraph("第十四部分 PPT 制作总建议与素材库", style="SectionTitle", page_break_before=True))
    paragraphs.append(mixed_paragraph([
        text_run("整体风格建议：", bold=True),
        text_run("采用“深蓝 + 科技蓝 + 橙色强调”的政企科技风，不建议做成互联网炫酷风，也不要做成纯学术报告。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("封面页建议：", bold=True),
        text_run("大标题一句话，副标题写“面向中国高速运营方的 AI 路网均衡与差异化收费决策平台”，背景图用高速路网夜景。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("图表页建议：", bold=True),
        text_run("所有数据图尽量只保留一个结论，不要一页放三张小图。政策页用时间线，财务页用柱状图加折线图，产品页用闭环流程图。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("字体建议：", bold=True),
        text_run("PPT 标题建议 28 至 32pt，正文 18 至 22pt，数字和核心结论加粗；一页不要超过 6 行正文。Word 文稿保持标题微软雅黑、正文宋体即可。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("页面节奏建议：", bold=True),
        text_run("前 3 页回答“为什么值得看”，中间 6 页回答“怎么做、怎么赚钱”，后 3 页回答“怎么落地、风险怎么控、要多少钱”。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("图片原则：", bold=True),
        text_run("优先使用官方政策页面截图、官方图解截图、可免费商用图库和自己绘制的图表，避免版权不清的随机网络图片。"),
    ]))
    paragraphs.append(mixed_paragraph([
        text_run("如果你时间很紧：", bold=True, color="C00000"),
        text_run("先把本文件中的 14 个部分一页一页搬进 PPT，再统一美化，比从零开始设计更高效。", color="C00000"),
    ]))
    paragraphs.append(plain_paragraph("素材与资料链接库", style="SubTitle"))
    link_items = [
        ("差异化收费实施方案", "https://xxgk.mot.gov.cn/2020/jigou/glj/202106/t20210615_3609815.html"),
        ("2023年交通运输行业发展统计公报", "https://xxgk.mot.gov.cn/2020/jigou/zhghs/202406/t20240614_4142419.html"),
        ("2021年全国收费公路统计公报", "https://xxgk.mot.gov.cn/jigou/glj/202211/t20221111_3707993.html"),
        ("公路水路交通基础设施数字化转型升级有关情况", "https://www.mot.gov.cn/2025wangshangzhibo/2025fourth/zhibozhaiyao/202504/t20250424_4167707.html"),
        ("公路水路交通基础设施数字化转型升级第三批公示", "https://xxgk.mot.gov.cn/jigou/zhghs/202604/t20260430_4204666.html"),
        ("湖北差异化收费政策页面", "https://www.wuhan.gov.cn/zwgk/tzgg/202306/t20230608_2213069.shtml"),
        ("广东差异化收费政策解读", "https://td.gd.gov.cn/zcwj_n/zcjd/content/mpost_4718830.html"),
        ("Pexels 高速公路图库", "https://www.pexels.com/search/highway/"),
        ("Pixabay 高速公路图库", "https://pixabay.com/images/search/highway/"),
        ("Iconfont 图标库", "https://www.iconfont.cn/"),
        ("Apache ECharts 图表示例", "https://echarts.apache.org/examples/zh/index.html"),
    ]
    for label, url in link_items:
        paragraphs.append(mixed_paragraph([link_run(label, url)], style="BulletText"))

    paragraphs.append(plain_paragraph("结语", style="SectionTitle", page_break_before=True))
    paragraphs.append(plain_paragraph("对外路演时，你们最值得坚持的一句话是：我们不是做“更贵的高速”，而是做“更聪明的高速运营”。在中国市场，这样的叙事更合规、更容易被客户接受，也更容易让投资人看懂你们的长期价值。"))
    paragraphs.append(plain_paragraph("如果后续还要继续完善，最值得补的两项材料是：一份带地区案例的试点测算表，以及一份更贴近真实采购口径的客户落地流程图。"))

    return paragraphs


def build_document_xml(paragraphs: Sequence[Paragraph]) -> tuple[str, str]:
    relationships: List[tuple[str, str]] = []
    rel_map: dict[str, str] = {}
    rel_index = 1

    def get_rel_id(url: str) -> str:
        nonlocal rel_index
        if url not in rel_map:
            rel_id = f"rId{rel_index}"
            rel_index += 1
            rel_map[url] = rel_id
            relationships.append((rel_id, url))
        return rel_map[url]

    body_parts: List[str] = []
    for paragraph in paragraphs:
        if paragraph.page_break_before:
            body_parts.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")
        ppr = [f'<w:pStyle w:val="{paragraph.style}"/>']
        if paragraph.align:
            ppr.append(f'<w:jc w:val="{paragraph.align}"/>')
        content_parts: List[str] = []
        for run in paragraph.runs:
            if isinstance(run, TextRun):
                content_parts.append(xml_run(run))
            else:
                rel_id = get_rel_id(run.url)
                text = escape(run.text)
                content_parts.append(
                    f'<w:hyperlink r:id="{rel_id}" w:history="1">'
                    "<w:r>"
                    "<w:rPr>"
                    '<w:rStyle w:val="Hyperlink"/>'
                    '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑" w:cs="Calibri"/>'
                    "</w:rPr>"
                    f'<w:t xml:space="preserve">{text}</w:t>'
                    "</w:r>"
                    "</w:hyperlink>"
                )
        body_parts.append(
            "<w:p>"
            f"<w:pPr>{''.join(ppr)}</w:pPr>"
            f"{''.join(content_parts)}"
            "</w:p>"
        )

    body_parts.append(
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1800" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="708"/>'
        '<w:docGrid w:linePitch="360"/>'
        "</w:sectPr>"
    )

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:body>
    {''.join(body_parts)}
  </w:body>
</w:document>
"""

    rels_xml_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Relationships xmlns="{PKG_REL_NS}">',
    ]
    for rel_id, url in relationships:
        rels_xml_parts.append(
            f'<Relationship Id="{rel_id}" Type="{EXT_REL_NS}" Target="{escape(url)}" TargetMode="External"/>'
        )
    rels_xml_parts.append("</Relationships>")
    return document_xml, "".join(rels_xml_parts)


def build_content_types_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CT_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def build_root_rels_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_core_xml() -> str:
    now = _dt.datetime(2026, 5, 5, 10, 0, 0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" xmlns:dcterms="{DCTERMS_NS}" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="{XSI_NS}">
  <dc:title>智能交通创业计划书-高速智价云</dc:title>
  <dc:subject>智能交通, 高速公路, 差异化收费, 创业计划书</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>智慧交通;高速公路;差异化收费;路演PPT</cp:keywords>
  <dc:description>面向中国市场的高速公路差异化收费与路网运营优化创业计划书。</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def build_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def write_docx(output_path: Path) -> None:
    paragraphs = build_content()
    document_xml, document_rels_xml = build_document_xml(paragraphs)
    styles_xml = build_styles_xml()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", build_content_types_xml())
        zf.writestr("_rels/.rels", build_root_rels_xml())
        zf.writestr("docProps/core.xml", build_core_xml())
        zf.writestr("docProps/app.xml", build_app_xml())
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)


def main() -> None:
    default_name = "智能交通创业计划书_高速智价云_路演底稿.docx"
    output_arg = os.environ.get("PLAN_DOCX_OUTPUT", "").strip()
    output_path = Path(output_arg) if output_arg else Path.cwd() / default_name
    write_docx(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
