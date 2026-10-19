n=0; ok=0
while IFS= read -r l; do n=$((n+1)); [ -f "/storage/$l" ] && ok=$((ok+1)); done < /tmp/newp.txt
m=0; ok2=0
while IFS= read -r l; do m=$((m+1)); [ -f "/storage/$l" ] && ok2=$((ok2+1)); done < /tmp/oldp.txt
echo "NEW $ok/$n  OLD $ok2/$m"
echo "files@Authors: $(find /storage/Authors -maxdepth 2 -type f 2>/dev/null | wc -l)  dirs@Authors: $(find /storage/Authors -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "empty dirs made 11:22: $(find /storage/Authors -maxdepth 1 -type d -newermt '2026-10-10 11:20' -empty | wc -l)"
