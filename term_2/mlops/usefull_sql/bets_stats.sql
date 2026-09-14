select
    bet_side,
    COUNT(*) as total_processed,
    COUNT(was_correct) as total_bets,
    SUM(CAST(was_correct as INT)) as correct_bets,
    SUM(CASE WHEN NOT was_correct THEN 1 ELSE 0 END) as incorrect_bets,
    (CAST(SUM(CAST(was_correct as INT)) as FLOAT) / COUNT(was_correct)) as bets_accuracy,
    SUM(CASE
        WHEN was_correct = true THEN bet_amount * (bet_return - 1)
        WHEN was_correct = false THEN -bet_amount
    END) as total_profit,
    AVG(CASE WHEN was_correct THEN bet_amount * (bet_return - 1) END) as avg_corrects_income,
    AVG(CASE WHEN NOT was_correct THEN -bet_amount END) as avg_incorrect_loss
from bets
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY bet_side;