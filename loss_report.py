"""
Generate loss analysis report for Telegram
"""
from loss_analyzer import LossAnalyzer
from datetime import datetime


def generate_loss_report() -> str:
    """
    Generate a comprehensive loss analysis report
    """
    loss_analyzer = LossAnalyzer()
    stats = loss_analyzer.get_failure_statistics()
    
    if stats['total_losses'] < 5:
        return "📊 Loss Analysis: Not enough data yet (need at least 5 losses)"
    
    report = f"""
📊 <b>LOSS ANALYSIS REPORT</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📉 <b>Total Losses Analyzed:</b> {stats['total_losses']}

🔍 <b>Most Common Failure:</b> {stats.get('most_common_failure', 'unknown')}

📋 <b>Failure Categories:</b>
"""
    
    for category, count in sorted(stats.get('failure_categories', {}).items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
        report += f"   • {category}: {count} losses\n"
    
    report += f"\n📚 <b>Top Lessons Learned:</b>\n"
    for i, lesson in enumerate(stats.get('top_lessons', [])[:5], 1):
        report += f"   {i}. {lesson}\n"
    
    # Generate prevention recommendations
    prevention_filters = loss_analyzer.generate_prevention_filters()
    if prevention_filters:
        report += f"\n🔧 <b>Recommended Filters:</b>\n"
        for filter_name, filter_data in prevention_filters.items():
            report += f"   • {filter_name}: {filter_data['current']:.3f} → {filter_data['suggested']:.3f}\n"
            report += f"     Reason: {filter_data['reason']}\n"
    
    report += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    report += f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return report

