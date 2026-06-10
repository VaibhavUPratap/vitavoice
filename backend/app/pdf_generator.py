import os
import datetime
import matplotlib
# Use non-interactive Agg backend to prevent Tkinter window issues in background threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def generate_shap_chart(shap_explanation, output_path):
    """
    Renders a horizontal SHAP value bar chart using Matplotlib and saves it as a PNG.
    """
    if not shap_explanation:
        return False
        
    labels = [feat['label'] for feat in shap_explanation]
    values = [feat['shap_value'] for feat in shap_explanation]
    
    # Reverse to keep the highest absolute value at the top of the horizontal bar chart
    labels.reverse()
    values.reverse()
    
    # Rose for increasing risk, Emerald for decreasing risk
    bar_colors = ['#e11d48' if val > 0 else '#059669' for val in values]
    
    fig, ax = plt.subplots(figsize=(6, 2.2))
    
    # Draw bars
    bars = ax.barh(labels, values, color=bar_colors, height=0.45, edgecolor='none')
    
    # Draw zero reference line
    ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#1e293b', labelsize=7.5)
    ax.set_title("Vocal Biomarker Classifier Contributions (SHAP Values)", fontsize=8.5, fontweight='bold', color='#0f172a', pad=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    return True

def generate_pdf_report(report_id, metrics, risk_score, confidence_calibration, shap_explanation, output_dir):
    """
    Generates a professional clinical screening PDF report.
    Returns the absolute path to the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = f"report_{report_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    # Temporary chart path
    temp_chart_path = os.path.join(output_dir, f"temp_{report_id}_shap.png")
    chart_created = generate_shap_chart(shap_explanation, temp_chart_path)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom medical styles
    style_title = ParagraphStyle(
        name='MedicalTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_LEFT
    )
    
    style_subtitle = ParagraphStyle(
        name='MedicalSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#0284c7'),
        alignment=TA_LEFT
    )
    
    style_section_h = ParagraphStyle(
        name='SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    style_body = ParagraphStyle(
        name='MedicalBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        alignment=TA_LEFT
    )
    
    style_meta_label = ParagraphStyle(
        name='MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=colors.HexColor('#475569')
    )
    
    style_meta_val = ParagraphStyle(
        name='MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#0f172a')
    )
    
    style_result_title = ParagraphStyle(
        name='ResultTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_CENTER
    )
    
    style_result_score = ParagraphStyle(
        name='ResultScore',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=26,
        textColor=colors.HexColor('#e11d48') if risk_score >= 0.65 else (colors.HexColor('#d97706') if risk_score >= 0.35 else colors.HexColor('#059669')),
        alignment=TA_CENTER
    )
    
    style_result_label = ParagraphStyle(
        name='ResultLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.HexColor('#475569'),
        alignment=TA_CENTER
    )
    
    style_disclaimer = ParagraphStyle(
        name='MedicalDisclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_JUSTIFY
    )

    story = []
    
    # 1. Header Section
    story.append(Paragraph("VITAVOICE HEALTHCARE", style_subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("AI-Assisted Vocal Screening Report", style_title))
    story.append(Spacer(1, 8))
    
    # Thin divider line
    d_table = Table([[""]], colWidths=[532], rowHeights=[1.5])
    d_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0284c7')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 10))
    
    # 2. Metadata Table
    meta_data = [
        [
            Paragraph("Report Reference:", style_meta_label),
            Paragraph(f"VV-{report_id}", style_meta_val),
            Paragraph("Screening Date:", style_meta_label),
            Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), style_meta_val)
        ],
        [
            Paragraph("System Model ID:", style_meta_label),
            Paragraph("VitaVoice-SVC-Ensemble-v2", style_meta_val),
            Paragraph("Environment Noise:", style_meta_label),
            Paragraph("Calibrated (Normal)", style_meta_val)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[110, 156, 110, 156])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # 3. Overall Screening Results Block & Clinical Summary
    risk_pct = f"{int(round(risk_score * 100))}%"
    risk_cat = "ELEVATED RISK" if risk_score >= 0.65 else ("BORDERLINE / MODERATE RISK" if risk_score >= 0.35 else "LOW RISK")
    cert_lbl = confidence_calibration.get('certainty_label', 'N/A')
    
    # Risk Box Color
    box_color = colors.HexColor('#fef2f2') if risk_score >= 0.65 else (colors.HexColor('#fffbeb') if risk_score >= 0.35 else colors.HexColor('#f0fdf4'))
    box_border = colors.HexColor('#fca5a5') if risk_score >= 0.65 else (colors.HexColor('#fcd34d') if risk_score >= 0.35 else colors.HexColor('#86efac'))
    
    res_box_data = [
        [Paragraph("Estimated Risk Score", style_result_title)],
        [Paragraph(risk_pct, style_result_score)],
        [Paragraph(f"<b>{risk_cat}</b>", style_result_title)],
        [Paragraph(cert_lbl, style_result_label)]
    ]
    res_box_table = Table(res_box_data, colWidths=[160])
    res_box_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), box_color),
        ('BOX', (0,0), (-1,-1), 1, box_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    
    # Clinical plain language summary text
    summary_text = (
        "Your vocal profile has been analyzed using standard acoustic biomarkers and deep speech foundation "
        "embeddings. Jitter, shimmer, and signal stability factors indicate a vocal pattern that correlates "
        "with "
    )
    if risk_score >= 0.65:
        summary_text += "neuromotor vocal tremors typically found in voice disorders or parkinsonian dysphonia."
    elif risk_score >= 0.35:
        summary_text += "minor speech irregularities, which can be indicative of vocal fatigue, temporary throat irritation, or early dysphonia stages."
    else:
        summary_text += "stable vocal fold vibrations with normal harmonic ratios, representing a healthy vocal profile."
    
    if shap_explanation and len(shap_explanation) > 0:
        top_biomarkers = [f"{f['label']} ({'increases risk' if f['shap_value'] > 0 else 'decreases risk'})" for f in shap_explanation[:3]]
        summary_text += f" AI explainability (SHAP) suggests the primary vocal drivers were {', '.join(top_biomarkers)}."
        
    summary_text += " This assessment is an automated pre-clinical screening aid. Please refer to recommendations below."
    
    summary_para = Paragraph(summary_text, style_body)
    
    # Table containing Risk Score Box (left) and Summary text (right)
    summary_layout_data = [
        [res_box_table, summary_para]
    ]
    summary_layout_table = Table(summary_layout_data, colWidths=[180, 352])
    summary_layout_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1,0), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(summary_layout_table)
    story.append(Spacer(1, 15))
    
    # 4. Vocal Biomarkers Details Table
    story.append(Paragraph("Acoustic Biomarker Analysis", style_section_h))
    
    jitter_threshold = 1.04
    shimmer_threshold = 3.80
    hnr_threshold = 20.0
    
    jitter_pct = metrics.get('jitter_pct', 0)
    shimmer_pct = metrics.get('shimmer_local', 0) * 100.0
    hnr = metrics.get('hnr', 0)
    f0 = metrics.get('fo_mean', 0)
    
    jitter_status = "Elevated" if jitter_pct > jitter_threshold else "Normal"
    shimmer_status = "Elevated" if shimmer_pct > shimmer_threshold else "Normal"
    hnr_status = "Low (Breathy)" if hnr < hnr_threshold else "Normal (Stable)"
    
    biomarkers_rows = [
        [
            Paragraph("<b>Biomarker</b>", style_meta_label),
            Paragraph("<b>Value</b>", style_meta_label),
            Paragraph("<b>Threshold</b>", style_meta_label),
            Paragraph("<b>Status</b>", style_meta_label),
            Paragraph("<b>Clinical Interpretation</b>", style_meta_label)
        ],
        [
            Paragraph("Average Pitch (F0)", style_body),
            Paragraph(f"{f0:.1f} Hz", style_body),
            Paragraph("85-255 Hz", style_body),
            Paragraph("Within Range" if f0 >= 80 and f0 <= 280 else "OutOfRange", style_body),
            Paragraph("Vocal cord oscillation speed.", style_body)
        ],
        [
            Paragraph("Pitch Jitter (Local)", style_body),
            Paragraph(f"{jitter_pct:.3f}%", style_body),
            Paragraph(f"< {jitter_threshold}%", style_body),
            Paragraph(jitter_status, style_body),
            Paragraph("Cycle-to-cycle frequency variations.", style_body)
        ],
        [
            Paragraph("Amplitude Shimmer", style_body),
            Paragraph(f"{shimmer_pct:.3f}%", style_body),
            Paragraph(f"< {shimmer_threshold}%", style_body),
            Paragraph(shimmer_status, style_body),
            Paragraph("Cycle-to-cycle amplitude variations.", style_body)
        ],
        [
            Paragraph("Harmonics-to-Noise (HNR)", style_body),
            Paragraph(f"{hnr:.1f} dB", style_body),
            Paragraph(f"> {hnr_threshold} dB", style_body),
            Paragraph(hnr_status, style_body),
            Paragraph("Vocal harmonics vs noise ratio.", style_body)
        ]
    ]
    
    biomarkers_table = Table(biomarkers_rows, colWidths=[120, 70, 75, 75, 192])
    biomarkers_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(biomarkers_table)
    story.append(Spacer(1, 15))
    
    # 5. SHAP Visual Chart (if created)
    chart_flowables = []
    if chart_created and os.path.exists(temp_chart_path):
        chart_img = Image(temp_chart_path, width=320, height=120)
        chart_flowables.append(chart_img)
        
    # Recommendations Paragraph
    rec_text = "<b>Recommended Action Plan:</b><br/>"
    if risk_score >= 0.65:
        rec_text += (
            "1. Schedule a formal diagnostic evaluation with a Neurologist or Otolaryngologist.<br/>"
            "2. Undergo a professional acoustic speech-language pathology assessment.<br/>"
            "3. Retest weekly in a calibrated quiet environment to track stability trends."
        )
    elif risk_score >= 0.35:
        rec_text += (
            "1. Rest your vocal folds, hydrate well, and avoid environmental allergens.<br/>"
            "2. Retest in 24-48 hours in a quiet environment to eliminate temporary noise biases.<br/>"
            "3. Monitor for other neuromotor symptoms (e.g. resting tremor, stiffness)."
        )
    else:
        rec_text += (
            "1. Maintain healthy vocal hygiene by drinking adequate fluids.<br/>"
            "2. Screen periodically (every 1-3 months) for wellness tracking.<br/>"
            "3. Consult a general physician if you develop persistent hoarseness."
        )
    rec_para = Paragraph(rec_text, style_body)
    
    # Combine Chart and Recommendations side-by-side or stacked
    if len(chart_flowables) > 0:
        clinical_details_data = [
            [chart_flowables[0], rec_para]
        ]
        clinical_details_table = Table(clinical_details_data, colWidths=[310, 222])
        clinical_details_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
            ('LEFTPADDING', (1,0), (1,0), 10),
        ]))
        story.append(clinical_details_table)
    else:
        story.append(rec_para)
        
    story.append(Spacer(1, 20))
    
    # 6. Medical Disclaimer block (Footer)
    disclaimer_border = Table([[""]], colWidths=[532], rowHeights=[0.5])
    disclaimer_border.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    
    disclaimer_text = (
        "<b>Responsible AI Screening Notice:</b> VitaVoice is an AI-powered pre-clinical voice pathology screening aid "
        "and is intended for educational, wellness tracking, and research purposes only. It is not an FDA-cleared diagnostic "
        "medical device. It does not provide medical diagnoses or treatment plans. Vocal irregularities can be caused by transient "
        "factors such as common cold, dehydration, allergies, or vocal strain. Always consult a qualified physician or specialist "
        "for definitive medical diagnoses and neurological evaluations."
    )
    disclaimer_para = Paragraph(disclaimer_text, style_disclaimer)
    
    story.append(KeepTogether([
        disclaimer_border,
        Spacer(1, 6),
        disclaimer_para
    ]))
    
    # Build Document
    doc.build(story)
    
    # Cleanup temporary Matplotlib image
    if chart_created and os.path.exists(temp_chart_path):
        try:
            os.remove(temp_chart_path)
        except Exception:
            pass
            
    return pdf_path
