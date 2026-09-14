import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import NotesPage from '../../pages/Notes';
import { AppData, Note, Area, SearchResult, Priority, Task } from '../../types';
import { api } from '../../services/api';

vi.mock('../../services/api');

const mockAppData: AppData = {
  user: { id: 1, full_name: 'Test User', email: 'test@example.com' },
  areas: [
    { id: 1, name: 'Work', color: 'bg-red-500' },
    { id: 2, name: 'Personal', color: 'bg-blue-500' },
  ],
  tasks: [ // Include some tasks for potential search results, even if not displayed
    { id: 't1', title: 'Task 1 for Work', area_id: 1, completed: false, description: '', due_date: '', priority: Priority.High, created_at: '' },
  ],
  notes: [
    { id: 'n1', title: 'Work Note 1', area_id: 1, content: 'Content for Work Note 1', created_at: '2026-04-10T10:00:00Z', updated_at: '2026-04-10T10:00:00Z' },
    { id: 'n2', title: 'Personal Note 1', area_id: 2, content: 'Content for Personal Note 1', created_at: '2026-04-11T10:00:00Z', updated_at: '2026-04-11T10:00:00Z' },
    { id: 'n3', title: 'Work Note 2', area_id: 1, content: 'Another work note', created_at: '2026-04-12T10:00:00Z', updated_at: '2026-04-12T10:00:00Z' },
  ],
};

const waitForDebounce = () => new Promise(resolve => setTimeout(resolve, 350));

describe('Notes Page', () => {
  const onOpenNote = vi.fn();
  const onNewNote = vi.fn();
  const onOpenTask = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers(); // Use fake timers for debounce testing
    (api.search as vi.Mock).mockResolvedValue([]); // Default mock for search
  });

  afterEach(() => {
    vi.useRealTimers(); // Restore real timers
  });

  const renderComponent = (data: AppData | null = mockAppData) => {
    render(
      <NotesPage
        data={data}
        onOpenNote={onOpenNote}
        onNewNote={onNewNote}
        onOpenTask={onOpenTask}
      />
    );
  };

  it('renders loading message when data is null', () => {
    renderComponent(null);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders notes correctly with all filter selected by default', () => {
    renderComponent();
    expect(screen.getByText('Notes')).toBeInTheDocument();
    expect(screen.getByText('All')).toHaveClass('bg-slate-900'); // Check if 'All' is active
    expect(screen.getByText('Work Note 1')).toBeInTheDocument();
    expect(screen.getByText('Personal Note 1')).toBeInTheDocument();
    expect(screen.getByText('Work Note 2')).toBeInTheDocument();
  });

  it('calls onNewNote when "New Note" header button is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByText('New Note', { selector: 'button' }));
    expect(onNewNote).toHaveBeenCalledTimes(1);
    expect(onNewNote).toHaveBeenCalledWith(null);
  });

  it('calls onOpenNote when a note card is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByText('Work Note 1'));
    expect(onOpenNote).toHaveBeenCalledTimes(1);
    expect(onOpenNote).toHaveBeenCalledWith(mockAppData.notes[0]);
  });

  it('filters notes by area when an area filter is clicked', async () => {
    renderComponent();
    
    // Click 'Work' area filter
    await userEvent.click(screen.getByRole('button', { name: 'Work' }));
    expect(screen.getByText('Work')).toHaveClass('bg-slate-900');
    expect(screen.getByText('Work Note 1')).toBeInTheDocument();
    expect(screen.queryByText('Personal Note 1')).not.toBeInTheDocument();
    expect(screen.getByText('Work Note 2')).toBeInTheDocument();

    // Click 'Personal' area filter
    await userEvent.click(screen.getByRole('button', { name: 'Personal' }));
    expect(screen.getByText('Personal')).toHaveClass('bg-slate-900');
    expect(screen.queryByText('Work Note 1')).not.toBeInTheDocument();
    expect(screen.getByText('Personal Note 1')).toBeInTheDocument();
    expect(screen.queryByText('Work Note 2')).not.toBeInTheDocument();

    // Click 'All' filter
    await userEvent.click(screen.getByRole('button', { name: 'All' }));
    expect(screen.getByText('All')).toHaveClass('bg-slate-900');
    expect(screen.getByText('Work Note 1')).toBeInTheDocument();
    expect(screen.getByText('Personal Note 1')).toBeInTheDocument();
    expect(screen.getByText('Work Note 2')).toBeInTheDocument();
  });

  it('handles search query and displays results', async () => {
    const searchResult: SearchResult[] = [{ id: 'n1', type: 'note', title: 'Work Note 1', content: 'Content for Work Note 1' }];
    (api.search as vi.Mock).mockResolvedValue(searchResult);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search notes...');
    
    await userEvent.type(searchInput, 'Work');
    vi.advanceTimersByTime(300); // Advance timers for debounce

    await waitFor(() => {
      expect(api.search).toHaveBeenCalledWith('Work', 'note');
      expect(screen.getByText('Work Note 1')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Work Note 1'));
    expect(onOpenNote).toHaveBeenCalledWith(mockAppData.notes[0]);
    expect(screen.queryByText('Work Note 1')).not.toBeInTheDocument(); // Search results should disappear
  });

  it('displays "No results found." when search returns empty', async () => {
    (api.search as vi.Mock).mockResolvedValue([]);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search notes...');
    
    await userEvent.type(searchInput, 'NonExistent');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(api.search).toHaveBeenCalledWith('NonExistent', 'note');
      expect(screen.getByText('No results found.')).toBeInTheDocument();
    });
  });

  it('clears search results and query on click outside', async () => {
    const searchResult: SearchResult[] = [{ id: 'n1', type: 'note', title: 'Work Note 1', content: 'Content for Work Note 1' }];
    (api.search as vi.Mock).mockResolvedValue(searchResult);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search notes...');
    
    await userEvent.type(searchInput, 'Work');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(screen.getByText('Work Note 1')).toBeInTheDocument();
    });

    // Simulate click outside the search area
    await userEvent.click(document.body);
    expect(screen.queryByText('Work Note 1')).not.toBeInTheDocument();
    expect(searchInput).toHaveValue('');
  });

  it('calls onNewNote when "Create New Note" placeholder is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: /Create New Note/i }));
    expect(onNewNote).toHaveBeenCalledTimes(1);
    expect(onNewNote).toHaveBeenCalledWith(undefined); // Called with no areaId
  });

  it('opens task when a task search result is returned (fallback)', async () => {
    const taskSearchResult: SearchResult[] = [{ id: 't1', type: 'task', title: 'Task 1 for Work' }];
    (api.search as vi.Mock).mockResolvedValue(taskSearchResult);
    
    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search notes...');
    await userEvent.type(searchInput, 'Task');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(screen.getByText('Task 1 for Work')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Task 1 for Work'));
    expect(onOpenTask).toHaveBeenCalledTimes(1);
    expect(onOpenTask).toHaveBeenCalledWith(mockAppData.tasks[0]);
    expect(screen.queryByText('Task 1 for Work')).not.toBeInTheDocument();
  });
});