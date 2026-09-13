use super::{Result, error, mapped};

// Check entry counts before zip0.6 allocates its file table. PyTorch emits ZIP64
// envelopes even for small PT2 archives, so both envelope forms are required.
pub(crate) fn validate_zip_envelope(
    bytes: &[u8],
    max_entries: usize,
    max_bytes: usize,
) -> Result<()> {
    if bytes.len() > max_bytes {
        return Err(error("ZIP envelope exceeds byte limit"));
    }
    fn u16_at(bytes: &[u8], offset: usize) -> Result<u64> {
        let data = bytes
            .get(
                offset
                    ..offset
                        .checked_add(2)
                        .ok_or_else(|| error("ZIP offset overflow"))?,
            )
            .ok_or_else(|| error("truncated ZIP header"))?;
        Ok(u16::from_le_bytes(data.try_into().expect("two bytes")) as u64)
    }
    fn u32_at(bytes: &[u8], offset: usize) -> Result<u64> {
        let data = bytes
            .get(
                offset
                    ..offset
                        .checked_add(4)
                        .ok_or_else(|| error("ZIP offset overflow"))?,
            )
            .ok_or_else(|| error("truncated ZIP header"))?;
        Ok(u32::from_le_bytes(data.try_into().expect("four bytes")) as u64)
    }
    fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
        let data = bytes
            .get(
                offset
                    ..offset
                        .checked_add(8)
                        .ok_or_else(|| error("ZIP offset overflow"))?,
            )
            .ok_or_else(|| error("truncated ZIP header"))?;
        Ok(u64::from_le_bytes(data.try_into().expect("eight bytes")))
    }
    let start = bytes.len().saturating_sub(65_557);
    let end = (start..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes.get(i..i + 4) == Some(b"PK\x05\x06"))
        .ok_or_else(|| error("missing bounded ZIP end record"))?;
    // Match zip 0.6's last-signature choice. Skipping a malformed later
    // signature could validate one envelope while the library parses another.
    if end + 22 + u16_at(bytes, end + 20)? as usize != bytes.len() {
        return Err(error("ZIP end record does not cover the complete envelope"));
    }
    if u16_at(bytes, end + 4)? != 0 || u16_at(bytes, end + 6)? != 0 {
        return Err(error("multi-disk ZIP is unsupported"));
    }
    let normal_disk_entries = u16_at(bytes, end + 8)?;
    let normal_entries = u16_at(bytes, end + 10)?;
    let normal_size = u32_at(bytes, end + 12)?;
    let normal_offset = u32_at(bytes, end + 16)?;
    let locator = end
        .checked_sub(20)
        .filter(|&i| bytes.get(i..i + 4) == Some(b"PK\x06\x07"));
    let (count, size, offset, boundary) = if let Some(locator) = locator {
        if u32_at(bytes, locator + 4)? != 0 || u32_at(bytes, locator + 16)? != 1 {
            return Err(error("multi-disk ZIP64 is unsupported"));
        }
        let position = usize::try_from(u64_at(bytes, locator + 8)?).map_err(mapped)?;
        if bytes.get(position..position.saturating_add(4)) != Some(b"PK\x06\x06") {
            return Err(error("invalid ZIP64 end pointer"));
        }
        let length = u64_at(bytes, position + 4)?;
        if length < 44
            || (position as u64)
                .checked_add(12)
                .and_then(|p| p.checked_add(length))
                != Some(locator as u64)
        {
            return Err(error("invalid ZIP64 record bounds"));
        }
        if u32_at(bytes, position + 16)? != 0 || u32_at(bytes, position + 20)? != 0 {
            return Err(error("multi-disk ZIP64 is unsupported"));
        }
        let disk_entries = u64_at(bytes, position + 24)?;
        let count = u64_at(bytes, position + 32)?;
        let size = u64_at(bytes, position + 40)?;
        let offset = u64_at(bytes, position + 48)?;
        if disk_entries != count
            || (normal_entries != u16::MAX as u64 && normal_entries != count)
            || (normal_disk_entries != u16::MAX as u64 && normal_disk_entries != count)
            || (normal_size != u32::MAX as u64 && normal_size != size)
            || (normal_offset != u32::MAX as u64 && normal_offset != offset)
        {
            return Err(error("inconsistent ZIP/ZIP64 directory metadata"));
        }
        (count, size, offset, position as u64)
    } else {
        if normal_entries == u16::MAX as u64
            || normal_size == u32::MAX as u64
            || normal_offset == u32::MAX as u64
            || normal_disk_entries != normal_entries
        {
            return Err(error("missing ZIP64 directory metadata"));
        }
        (normal_entries, normal_size, normal_offset, end as u64)
    };
    if count == 0
        || count > max_entries as u64
        || size > max_bytes as u64
        || offset.checked_add(size) != Some(boundary)
        || size < count.saturating_mul(46)
    {
        return Err(error("ZIP directory count or bounds exceed limits"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    use zip::{ZipArchive, ZipWriter, write::FileOptions};

    fn archive() -> Vec<u8> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        writer.start_file("state", FileOptions::default()).unwrap();
        writer.write_all(b"model").unwrap();
        writer.finish().unwrap().into_inner()
    }

    fn zip64(bytes: &[u8], count: u64) -> Vec<u8> {
        let end = bytes.len() - 22;
        let mut result = bytes[..end].to_vec();
        let size = u32::from_le_bytes(bytes[end + 12..end + 16].try_into().unwrap());
        let offset = u32::from_le_bytes(bytes[end + 16..end + 20].try_into().unwrap());
        result.extend_from_slice(b"PK\x06\x06");
        result.extend_from_slice(&44_u64.to_le_bytes());
        result.extend_from_slice(&45_u16.to_le_bytes());
        result.extend_from_slice(&45_u16.to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());
        result.extend_from_slice(&count.to_le_bytes());
        result.extend_from_slice(&count.to_le_bytes());
        result.extend_from_slice(&(size as u64).to_le_bytes());
        result.extend_from_slice(&(offset as u64).to_le_bytes());
        result.extend_from_slice(b"PK\x06\x07");
        result.extend_from_slice(&0_u32.to_le_bytes());
        result.extend_from_slice(&(end as u64).to_le_bytes());
        result.extend_from_slice(&1_u32.to_le_bytes());
        let mut normal = bytes[end..].to_vec();
        normal[8..12].fill(0xff);
        normal[12..20].fill(0xff);
        result.extend_from_slice(&normal);
        result
    }

    #[test]
    fn zip_envelope_bounds_counts_before_native_allocation() {
        let bytes = archive();
        validate_zip_envelope(&bytes, 1, 1024).unwrap();
        let normal_end = bytes.len() - 22;
        let mut forged = bytes.clone();
        forged[normal_end + 8..normal_end + 12].copy_from_slice(&[0xfe, 0xff, 0xfe, 0xff]);
        assert!(validate_zip_envelope(&forged, 3, 1024).is_err());
        let large = zip64(&bytes, u64::MAX);
        assert!(validate_zip_envelope(&large, 3, 1024).is_err());
        assert!(validate_zip_envelope(&zip64(&bytes, 10_001), 10_000, 1024).is_err());
        let valid = zip64(&bytes, 1);
        validate_zip_envelope(&valid, 1, 1024).unwrap();
        assert_eq!(ZipArchive::new(Cursor::new(valid)).unwrap().len(), 1);
    }

    #[test]
    fn zip_envelope_rejects_conflicting_records_and_out_of_bounds_offsets() {
        let bytes = archive();
        let valid = zip64(&bytes, 1);
        let footer = valid.len() - 22;
        for case in 0..5 {
            let mut invalid = valid.clone();
            match case {
                0 => invalid[footer + 20..footer + 22].copy_from_slice(&1_u16.to_le_bytes()),
                1 => invalid[footer + 8..footer + 10].copy_from_slice(&2_u16.to_le_bytes()),
                2 => invalid[footer - 12..footer - 4].copy_from_slice(&u64::MAX.to_le_bytes()),
                3 => invalid[footer - 4..footer].copy_from_slice(&2_u32.to_le_bytes()),
                _ => invalid[bytes.len() - 22 + 40..bytes.len() - 22 + 48]
                    .copy_from_slice(&u64::MAX.to_le_bytes()),
            }
            assert!(
                validate_zip_envelope(&invalid, 3, 1024).is_err(),
                "case {case}"
            );
        }
        // The library chooses this later signature even though its comment
        // length is inconsistent; preflight must not fall back to the first.
        let mut invalid = bytes.clone();
        let end = invalid.len() - 22;
        invalid[end + 20..end + 22].copy_from_slice(&22_u16.to_le_bytes());
        let mut later = bytes[end..].to_vec();
        later[20..22].copy_from_slice(&1_u16.to_le_bytes());
        invalid.extend_from_slice(&later);
        assert!(validate_zip_envelope(&invalid, 3, 1024).is_err());
        assert!(validate_zip_envelope(&bytes, 1, bytes.len() - 1).is_err());
    }
}
