"""Render the received-corpus flow from audited counts; no literature inference."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(bundle, output, preview=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    flow = json.loads((bundle / 'received_flow.json').read_text())
    d = flow['received_group_final_dispositions']
    excluded_screen = d['excluded_at_title_or_abstract'] + d['excluded_after_metadata_or_abstract']
    contexts_screen = d['context_candidate']
    retained = 9559 - excluded_screen - contexts_screen
    primary_groups = retained - d['report_not_retrieved'] - d['container_linked_existing_report']
    context_primary = d['context_primary_source'] + d['context_criteria_unidentified']
    assert excluded_screen + contexts_screen + retained == 9559
    assert primary_groups == sum(d[k] for k in (
        'included_applied_narrative', 'excluded_after_primary_sections',
        'context_primary_source', 'context_criteria_unidentified',
        'eligibility_unresolved_after_primary_sections'))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(.5, .975, 'Flujo del corpus recibido', ha='center', va='top',
            fontsize=19, fontweight='bold', color='#17324d')

    def box(x, y, label, color='#edf4fa', width=.29, height=.077):
        patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                               boxstyle='round,pad=0.009,rounding_size=0.008',
                               facecolor=color, edgecolor='#58738a', linewidth=1)
        ax.add_patch(patch)
        ax.text(x, y, label, ha='center', va='center', fontsize=10.5,
                color='#17324d', linespacing=1.4)

    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a,
                    arrowprops={'arrowstyle':'->', 'color':'#58738a', 'lw':1.1})

    box(.36, .86, '11.635 ocurrencias recibidas\nTres rondas documentadas', width=.40)
    box(.80, .86, 'Duplicados: 1.720 internos\n+ 356 enlaces entre rondas', '#f3f4f5', width=.30)
    arrow((.57, .86), (.64, .86))
    box(.36, .70, '9.559 grupos cribados\nTítulo, resumen o evidencia previa', width=.40)
    arrow((.36, .813), (.36, .748))
    box(.80, .70, format(excluded_screen, ',').replace(',', '.') + ' fuera del núcleo\npor título, resumen o metadatos',
        '#f3f4f5', width=.30)
    arrow((.57, .70), (.64, .70))
    box(.80, .56, f'{contexts_screen:,} candidatos contextuales\nNo inclusiones aplicadas'.replace(',', '.'),
        '#f3f4f5', width=.30)
    arrow((.36, .653), (.36, .56))
    arrow((.36, .56), (.64, .56))
    box(.36, .46, f'{retained} grupos retenidos\npara comprobación documental', width=.40)
    arrow((.36, .56), (.36, .508))
    box(.80, .42, f'{d["report_not_retrieved"]} registros sin informe\nElegibilidad desconocida\n{d["container_linked_existing_report"]} contenedores vinculados',
        '#fff6e7', width=.30, height=.095)
    arrow((.57, .46), (.64, .43))
    box(.36, .30, f'{primary_groups} grupos con evidencia primaria\n{flow["primary_reports_assessed_or_reused"]} evaluaciones focales o reutilizadas',
        width=.40, height=.085)
    arrow((.36, .413), (.36, .35))
    box(.17, .13, f'{d["excluded_after_primary_sections"]} excluidos tras\nsecciones primarias',
        '#f3f4f5', width=.28)
    box(.50, .13, f'{d["included_applied_narrative"]} grupos / '
        f'{flow["unique_studies_included_applied_narrative"]} estudios\nInclusión narrativa aplicada',
        '#e8f4ed', width=.30)
    box(.83, .13, f'{context_primary} contextos primarios\n'
        f'{d["eligibility_unresolved_after_primary_sections"]} criterios DL no resueltos',
        '#edf4fa', width=.28)
    for x in (.17, .50, .83):
        arrow((.36, .248), (x, .18))
    ax.text(.5, .035,
            'Los alias conservan los grupos y evitan duplicar estudios.\n'
            'Cobertura histórica limitada: 374 Scopus no recibidos; totales/rangos WoS no confirmados.',
            ha='center', va='center', fontsize=9, color='#44596a', linespacing=1.4)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'svg.hashsalt':'e2-received-prisma-v1'}):
        fig.savefig(output / 'prisma_received.svg', bbox_inches='tight',
                    metadata={'Date':None, 'Creator':'E2 audited flow renderer'})
    svg = output / 'prisma_received.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    if preview:
        fig.savefig(preview, bbox_inches='tight', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path,
                        default=ROOT/'references/literature_e2/prisma_received_v1')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'book_e2/_static/e2_prisma_received_v1')
    parser.add_argument('--preview-png', type=Path)
    args = parser.parse_args()
    render(args.bundle, args.output_dir, args.preview_png)
