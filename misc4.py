# FLAG CHECKING FOR STEADY STATE
flags = []
reasons = []

# Round to 4 decimal places for comparison (adjust as needed)
ROUND_DIGITS = 4

# Use dynamic thresholds if provided
if dynamic_thresholds:
    # Use dynamic thresholds for this OFP/test_case group
    group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
    if group_key in dynamic_thresholds:
        thresh = dynamic_thresholds[group_key]
        
        # Check all 4 thresholds
        failed_checks = 0
        
        # Check variance - round before comparing
        if 'min_variance' in thresh and 'max_variance' in thresh:
            var_rounded = round(metrics['variance'], ROUND_DIGITS)
            min_thresh_rounded = round(thresh['min_variance'], ROUND_DIGITS)
            max_thresh_rounded = round(thresh['max_variance'], ROUND_DIGITS)
            
            if var_rounded < min_thresh_rounded:
                failed_checks += 1
                reasons.append(f"Variance {metrics['variance']:.3f} < {thresh['min_variance']:.3f} (dynamic)")
            elif var_rounded > max_thresh_rounded:
                failed_checks += 1
                reasons.append(f"Variance {metrics['variance']:.3f} > {thresh['max_variance']:.3f} (dynamic)")
        
        # Check std - round before comparing
        if 'min_std' in thresh and 'max_std' in thresh:
            std_rounded = round(metrics['std'], ROUND_DIGITS)
            min_thresh_rounded = round(thresh['min_std'], ROUND_DIGITS)
            max_thresh_rounded = round(thresh['max_std'], ROUND_DIGITS)
            
            if std_rounded < min_thresh_rounded:
                failed_checks += 1
                reasons.append(f"Std {metrics['std']:.3f} < {thresh['min_std']:.3f} (dynamic)")
            elif std_rounded > max_thresh_rounded:
                failed_checks += 1
                reasons.append(f"Std {metrics['std']:.3f} > {thresh['max_std']:.3f} (dynamic)")
        
        # Check slope - round before comparing
        if 'min_slope' in thresh and 'max_slope' in thresh:
            slope_rounded = round(metrics['slope'], ROUND_DIGITS)
            min_thresh_rounded = round(thresh['min_slope'], ROUND_DIGITS)
            max_thresh_rounded = round(thresh['max_slope'], ROUND_DIGITS)
            
            if slope_rounded < min_thresh_rounded or slope_rounded > max_thresh_rounded:
                failed_checks += 1
                if slope_rounded < min_thresh_rounded:
                    reasons.append(f"Slope {metrics['slope']:.4f} < {thresh['min_slope']:.4f} (dynamic)")
                else:
                    reasons.append(f"Slope {metrics['slope']:.4f} > {thresh['max_slope']:.4f} (dynamic)")
        
        # Check IQR - round before comparing
        if 'min_iqr' in thresh and 'max_iqr' in thresh:
            iqr_rounded = round(metrics['iqr'], ROUND_DIGITS)
            min_thresh_rounded = round(thresh['min_iqr'], ROUND_DIGITS)
            max_thresh_rounded = round(thresh['max_iqr'], ROUND_DIGITS)
            
            if iqr_rounded < min_thresh_rounded:
                failed_checks += 1
                reasons.append(f"IQR {metrics['iqr']:.3f} < {thresh['min_iqr']:.3f} (dynamic)")
            elif iqr_rounded > max_thresh_rounded:
                failed_checks += 1
                reasons.append(f"IQR {metrics['iqr']:.3f} > {thresh['max_iqr']:.3f} (dynamic)")
        
        # Only flag if ALL 4 fail
        if failed_checks == 4:
            flags.append('all_thresholds_failed')
        else:
            reasons = []  # Clear reasons if not flagging
